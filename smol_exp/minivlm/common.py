"""Shared code of the mini-VLM experiment: four pre-trainings of a ViT encoder, same recipe.

A mini-VLM = frozen vision encoder -> connector (2x2 pixel shuffle + MLP, trained) -> SmolLM2-135M-Instruct
(LoRA in stage 2). The four encoders are ViTs with patch 14 at 224 px (256 patch tokens), so the main
difference is how they were pre-trained:
  siglip: contrastive image-text, sigmoid loss       clip: contrastive image-text, softmax loss
  ijepa:  predicts hidden parts in feature space     mae:  reconstructs hidden pixels
(I-JEPA and MAE are both ViT-H/14: the cleanest pair of the four.)

The images of the attack and of the evaluation are exactly those of the SmolVLM notebooks
(01_train_universal_perturbation.ipynb): ImageNetV2 matched-frequency, sorted file list,
np.random.RandomState(0).permutation, first 500 = build set, next 500 = held-out set.
"""
import glob
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('MINIVLM_DATA', '/Data/gurvan.nkouathio-kemnhou/minivlm'))
RESULTS_DIR = Path(os.environ.get('MINIVLM_RESULTS', HERE / 'results'))

ENCODERS = {
    # name: Hugging Face id (all: ViT, patch 14, 224 x 224 -> 16 x 16 = 256 patch tokens)
    'siglip': 'google/siglip-so400m-patch14-224',   # SO400M (~400M), contrastive image-text (sigmoid), WebLI
    'clip': 'openai/clip-vit-large-patch14',        # ViT-L/14 (~300M), contrastive image-text (softmax), 400M pairs
    'ijepa': 'facebook/ijepa_vith14_1k',            # ViT-H/14 (~630M), predictive in feature space, ImageNet-1k
    'mae': 'facebook/vit-mae-huge',                 # ViT-H/14 (~630M), pixel reconstruction, ImageNet-1k
}
LM_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'      # the language model of SmolVLM-256M
IMG_SIZE = 224
PROMPT = 'Describe this image in one sentence.'     # same prompt as the SmolVLM notebooks
MAX_NEW_TOKENS = 40
IMAGE_MARKER = '<IMAGE>'

# Same split as the SmolVLM notebooks
SEED = 0
N_BUILD = 500
N_EVAL = 500
XI = 10 / 255
CAPTION_SIM_CHANGED = 0.5    # and 0.7 as the stricter check of the report


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def amp_dtype():
    # bfloat16 when the GPU supports it, float16 otherwise (with a GradScaler for training)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# A mini-VLM whose encoder was ALSO trained during the post-training (stage 2), as in SmolVLM: '<encoder>-unfrozen'.
# Its fine-tuned encoder weights stay on the local disk (unfrozen_encoder_path), its connector and LoRA in results/.
UNFROZEN = '-unfrozen'
MODELS = list(ENCODERS) + [e + UNFROZEN for e in ENCODERS]


def base_encoder(name):
    return name[:-len(UNFROZEN)] if name.endswith(UNFROZEN) else name


def unfrozen_encoder_path(name):
    return DATA_DIR / 'unfrozen' / ('%s_encoder.pt' % base_encoder(name))


def other_encoders(name):
    """Perturbations tested as transfers on this model. For an unfrozen model: the perturbation of the ORIGINAL
    encoder (built on the frozen mini-VLM), i.e. the fixed attack, not recomputed against the post-trained encoder."""
    if name.endswith(UNFROZEN):
        return [base_encoder(name)]
    return [e for e in ENCODERS if e != name]


def results_dir(encoder):
    d = RESULTS_DIR / encoder
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------- images

PREPROC = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE), transforms.PILToTensor()])


def load_u8(path):
    return PREPROC(Image.open(path).convert('RGB'))


def imagenetv2_split():
    """Build and held-out files, in the same order as the SmolVLM notebooks."""
    files = sorted(glob.glob(str(DATA_DIR / 'imagenetv2' / 'imagenetv2-*' / '*' / '*.jpeg')))
    assert len(files) == 10000, 'ImageNetV2 not found or incomplete in %s (%d images): run prepare_data.sh' % (
        DATA_DIR / 'imagenetv2', len(files))
    perm = np.random.RandomState(SEED).permutation(len(files))
    build = [files[i] for i in perm[:N_BUILD]]
    held_out = [files[i] for i in perm[N_BUILD:N_BUILD + N_EVAL]]
    return build, held_out


def imagenetv2_images():
    """(build, held_out) uint8 tensors N x 3 x 224 x 224, cached on the local disk, plus the file names."""
    build_files, eval_files = imagenetv2_split()
    cache = DATA_DIR / 'imagenetv2_split_224.pt'
    if cache.exists():
        d = torch.load(cache)
    else:
        d = {'build': torch.stack([load_u8(f) for f in build_files]),
             'eval': torch.stack([load_u8(f) for f in eval_files])}
        torch.save(d, cache)
    rel = lambda fs: [os.path.relpath(f, DATA_DIR / 'imagenetv2') for f in fs]
    return d['build'], d['eval'], rel(build_files), rel(eval_files)


class CocoCaptions(torch.utils.data.Dataset):
    """COCO 2017 images (224 x 224 uint8) with their 5 captions."""

    def __init__(self, split='train2017', max_images=None):
        root = DATA_DIR / 'coco'
        ann = json.load(open(root / 'annotations' / ('captions_%s.json' % split)))
        files = {im['id']: im['file_name'] for im in ann['images']}
        caps = defaultdict(list)
        for a in ann['annotations']:
            caps[a['image_id']].append(a['caption'].strip())
        ids = sorted(caps)[:max_images]
        self.ids = ids
        self.items = [(str(root / split / files[i]), caps[i]) for i in ids]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return load_u8(self.items[i][0]), i


def quantize(x01):
    # The perturbed image is saved as an ordinary 8-bit image, as in the "real pipeline" of the report
    return torch.round(x01.clamp(0, 1) * 255) / 255


# ----------------------------------------------------------------------------------------------- models

class Encoder(nn.Module):
    """Frozen vision encoder: [0, 1] images 224 x 224 -> patch tokens B x 256 x dim (last layer).
    The CLS token of CLIP and MAE is dropped, so that every encoder gives the same 256 patch tokens."""

    def __init__(self, name):
        super().__init__()
        from transformers import AutoImageProcessor
        trained, name = name.endswith(UNFROZEN), base_encoder(name)
        hf_id = ENCODERS[name]
        self.name = name
        if name == 'siglip':
            from transformers import SiglipVisionModel
            self.net = SiglipVisionModel.from_pretrained(hf_id)
        elif name == 'clip':
            from transformers import CLIPVisionModel
            self.net = CLIPVisionModel.from_pretrained(hf_id)
        elif name == 'mae':
            # No masking at inference (the pre-training masks 75% of the patches)
            from transformers import ViTMAEModel
            self.net = ViTMAEModel.from_pretrained(hf_id, mask_ratio=0.0)
        else:
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained(hf_id)
        ip = AutoImageProcessor.from_pretrained(hf_id)
        self.register_buffer('mean', torch.tensor(ip.image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(ip.image_std).view(1, 3, 1, 1))
        self.dim = self.net.config.hidden_size
        if trained:   # encoder post-trained with the connector and the LM (train_connector.py on '<name>-unfrozen')
            path = unfrozen_encoder_path(name)
            assert path.exists(), '%s not found: train it first on THIS machine (local disk)' % path
            self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x01):
        kw = {}
        if self.name == 'mae':
            # ViTMAE shuffles the patches with random noise even when nothing is masked: an increasing noise
            # keeps them in their original order (checked: ids_restore is then the identity)
            kw['noise'] = torch.arange(256, device=x01.device, dtype=torch.float32).expand(x01.shape[0], -1)
        tokens = self.net(pixel_values=(x01 - self.mean) / self.std, **kw).last_hidden_state
        if tokens.shape[1] == 257:
            tokens = tokens[:, 1:]          # CLS token
        assert tokens.shape[1] == 256, 'expected 256 patch tokens, got %d' % tokens.shape[1]
        return tokens


class Connector(nn.Module):
    """As in SmolVLM: 2 x 2 pixel shuffle (256 tokens -> 64 tokens of 4 x dim), then a 2-layer MLP to the LM width.
    A LayerNorm first, so that both encoders reach the MLP at the same scale."""

    def __init__(self, enc_dim, lm_dim, scale=2):
        super().__init__()
        self.scale = scale
        self.norm = nn.LayerNorm(enc_dim)
        self.mlp = nn.Sequential(nn.Linear(enc_dim * scale * scale, lm_dim), nn.GELU(), nn.Linear(lm_dim, lm_dim))

    def forward(self, tokens):
        b, n, d = tokens.shape
        s, r = int(round(math.sqrt(n))), self.scale
        x = self.norm(tokens).view(b, s // r, r, s // r, r, d).permute(0, 1, 3, 2, 4, 5).reshape(b, (s // r) ** 2, r * r * d)
        return self.mlp(x)


def load_lm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LM_ID)
    lm = AutoModelForCausalLM.from_pretrained(LM_ID, torch_dtype=torch.float32)
    return lm, tok


def add_lora(lm, r=16):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type='CAUSAL_LM',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    return get_peft_model(lm, cfg)


class MiniVLM(nn.Module):
    """Prompt = chat template of SmolLM2 with the 64 image embeddings in place of IMAGE_MARKER.
    Every image gets exactly the same prompt, so batches need no padding before the answer."""

    def __init__(self, encoder, connector, lm, tok):
        super().__init__()
        self.encoder, self.connector, self.lm, self.tok = encoder, connector, lm, tok
        text = tok.apply_chat_template([{'role': 'user', 'content': IMAGE_MARKER + '\n' + PROMPT}],
                                       tokenize=False, add_generation_prompt=True)
        pre, post = text.split(IMAGE_MARKER)
        self.register_buffer('pre_ids', torch.tensor(tok(pre, add_special_tokens=False).input_ids), persistent=False)
        self.register_buffer('post_ids', torch.tensor(tok(post, add_special_tokens=False).input_ids), persistent=False)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def embed(self, ids):
        return self.lm.get_input_embeddings()(ids)

    def prefix(self, tokens):
        """Encoder tokens B x 256 x dim -> prompt embeddings B x L x lm_dim."""
        img = self.connector(tokens)
        b = img.shape[0]
        pre = self.embed(self.pre_ids).unsqueeze(0).expand(b, -1, -1)
        post = self.embed(self.post_ids).unsqueeze(0).expand(b, -1, -1)
        return torch.cat([pre, img.to(pre.dtype), post], dim=1)

    def caption_loss(self, tokens, captions, max_len=48, reduction='mean'):
        """Cross-entropy of the captions (followed by the end-of-turn token), the prompt is not scored.
        reduction='none': mean cross-entropy of each caption -> (B,)"""
        prefix = self.prefix(tokens)
        b, lp, _ = prefix.shape
        enc = [self.tok(c + self.tok.eos_token, add_special_tokens=False).input_ids[:max_len] for c in captions]
        lt = max(len(e) for e in enc)
        ids = torch.full((b, lt), self.pad_id, dtype=torch.long, device=prefix.device)
        mask = torch.zeros((b, lt), dtype=torch.long, device=prefix.device)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e, device=prefix.device)
            mask[i, :len(e)] = 1
        embeds = torch.cat([prefix, self.embed(ids)], dim=1)
        attention = torch.cat([torch.ones((b, lp), dtype=torch.long, device=prefix.device), mask], dim=1)
        # Same loss as lm(..., labels=...), but the vocabulary logits (49k words) are only computed at the
        # positions that predict a caption token, not on the whole prompt: much less GPU memory
        base = self.lm.get_base_model() if hasattr(self.lm, 'get_base_model') else self.lm
        hidden = base.get_decoder()(inputs_embeds=embeds, attention_mask=attention).last_hidden_state
        logits = base.get_output_embeddings()(hidden[:, lp - 1:lp - 1 + lt]).float()
        if reduction == 'none':
            ce = F.cross_entropy(logits.transpose(1, 2), ids, reduction='none')        # B x lt
            return (ce * mask).sum(1) / mask.sum(1)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.masked_fill(mask == 0, -100).reshape(-1),
                               ignore_index=-100)

    @torch.no_grad()
    def generate_from_tokens(self, tokens, max_new_tokens=MAX_NEW_TOKENS):
        prefix = self.prefix(tokens)
        attention = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
        out = self.lm.generate(inputs_embeds=prefix, attention_mask=attention, max_new_tokens=max_new_tokens,
                               do_sample=False, pad_token_id=self.pad_id, eos_token_id=self.tok.eos_token_id)
        # With inputs_embeds only, generate returns the new tokens only
        return [s.strip() for s in self.tok.batch_decode(out, skip_special_tokens=True)]

    @torch.no_grad()
    def caption(self, x01, batch_size=32):
        caps = []
        for i in range(0, len(x01), batch_size):
            with torch.autocast('cuda', dtype=amp_dtype(), enabled=x01.is_cuda):
                tokens = self.encoder(x01[i:i + batch_size])
                caps += self.generate_from_tokens(tokens.float())
        return caps


def load_minivlm(encoder_name, device, stage=2):
    """Trained mini-VLM (connector of the given stage, and LoRA for stage 2)."""
    d = results_dir(encoder_name)
    encoder = Encoder(encoder_name)
    lm, tok = load_lm()
    connector = Connector(encoder.dim, lm.config.hidden_size)
    connector.load_state_dict(torch.load(d / ('connector_stage%d.pt' % stage), map_location='cpu'))
    if stage == 2:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, str(d / 'lora_stage2'))
    vlm = MiniVLM(encoder, connector, lm, tok).to(device).eval()
    return vlm


# ----------------------------------------------------------------------------------------------- metrics

def token_cosine(a, b):
    """Mean over tokens of the per-token cosine similarity -> (B,)"""
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean(dim=1)


class TextSim:
    """Caption similarity of the report: MiniLM sentence embeddings, cosine."""

    def __init__(self, device):
        from sentence_transformers import SentenceTransformer
        self.st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=str(device))

    def __call__(self, a, b):
        ea = self.st.encode(a, convert_to_tensor=True, normalize_embeddings=True)
        eb = self.st.encode(b, convert_to_tensor=True, normalize_embeddings=True)
        return (ea * eb).sum(-1).cpu()


def wilson(k, n, z=1.96):
    """95% Wilson interval of k successes out of n, in %."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load_perturbation(path, device):
    """A perturbation saved by uap_encoder.py (.pt, 1 x 3 x 224 x 224) or by the SmolVLM notebooks
    (.npy, 1 x 3 x 512 x 512, resized to 224). Values on the [0, 1] scale."""
    path = str(path)
    v = torch.from_numpy(np.load(path)) if path.endswith('.npy') else torch.load(path, map_location='cpu')
    v = v.float().reshape(1, 3, v.shape[-2], v.shape[-1])
    if v.shape[-1] != IMG_SIZE:
        xi = v.abs().max()
        v = F.interpolate(v, size=IMG_SIZE, mode='bilinear', antialias=True, align_corners=False).clamp(-xi, xi)
    return v.to(device)


# ----------------------------------------------------------------------------------------------- seeds and controls

N_RANDOM_DRAWS = 5   # random ±xi controls: draw 0 (the original 'random ±xi') + draws 1..4


def seed_tag(seed):
    """File suffix of a perturbation built with another seed (seed 0 = the original files, no suffix)."""
    return '' if seed == 0 else '_s%d' % seed


def seed_start(seed, size, xi):
    """Starting point of the attack: zero for seed 0 (original runs), uniform in the l_inf ball otherwise, so that
    runs with different seeds differ by their start AND by the order of the build images."""
    if seed == 0:
        return None
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 3, size, size, generator=g) * 2 - 1) * xi


def random_draw(k, size, xi, device):
    """Random ±xi sign pattern number k >= 1 (draw 0 is the original 'random ±xi', made with the global seed)."""
    g = torch.Generator().manual_seed(10_000 + k)
    return (xi * torch.sign(torch.randn(1, 3, size, size, generator=g))).to(device)


def extra_conditions(out, xi_255, size, device, own_name):
    """Perturbations added to every evaluation: own v built with other seeds (uap_xi10_s<k>.pt, if computed) and
    random draws 1..N_RANDOM_DRAWS-1. Returns {name: path or tensor}."""
    conds = {}
    for f in sorted(out.glob('uap_xi%g_s*.pt' % xi_255)):
        if f.stem.endswith('_ckpt'):
            continue
        conds['%s, seed %s' % (own_name, f.stem.split('_s')[-1])] = f
    for k in range(1, N_RANDOM_DRAWS):
        conds['random ±xi, draw %d' % k] = random_draw(k, size, xi_255 / 255, device)
    return conds


# ----------------------------------------------------------------------------------------------- attack to convergence

@torch.no_grad()
def features_of(feat_fn, images_u8, device, v=None, bs=32):
    """Features (tokens or connector outputs) of [0, 255] images, perturbed images rounded to 8 bits, in bf16."""
    out = []
    for i in range(0, len(images_u8), bs):
        x = images_u8[i:i + bs].to(device).float() / 255
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out.append(feat_fn(x if v is None else quantize(x + v)).to(torch.bfloat16))
    return torch.cat(out)


def feature_cosine(feat_fn, images_u8, device, clean, v=None, bs=32):
    """Per-image token cosine between `clean` features and those of the perturbed images, batch by batch (never holds
    all perturbed features: 500 images of Qwen2-VL are 1024 tokens x 1280 dims each) -> (N,) on the CPU."""
    out = []
    with torch.no_grad():
        for i in range(0, len(images_u8), bs):
            z = features_of(feat_fn, images_u8[i:i + bs], device, v, bs)
            out.append(token_cosine(clean[i:i + bs].to(z.device), z).cpu())
    return torch.cat(out)


def converge_uap(feat_fn, build_u8, device, v0=None, xi=XI, lr=1 / 255, batch_size=8, ckpt_path=None, min_epochs=10,
                 max_epochs=60, min_delta=1e-3, plateau=2, max_halvings=3, name='uap', score_fn=None, eval_bs=32,
                 seed=SEED):
    """Label-free universal perturbation, run to CONVERGENCE (same loss and optimiser as the SmolVLM notebooks:
    mean cosine between clean and perturbed features, Adam, projection on the l_inf ball).

    After each epoch, v is scored on all the build images (rounded to 8 bits, as in the evaluation; lower = stronger)
    and the best v is kept. If the score has not improved by min_delta for `plateau` epochs, the optimisation restarts
    from the best v with half the step. It stops after max_halvings halvings without improvement (or max_epochs),
    never before min_epochs (the 10 epochs of the report). Only build images are used; resumable with ckpt_path.
    score_fn(x01, idx) -> (B,) replaces the feature cosine (lower = stronger attack, differentiable), e.g. the
    likelihood of the clean captions (end-to-end attack); x01 are the perturbed images of the build images idx.
    seed: order of the build images (seed 0 = the original runs); a v0 can be given for a random start (seed_start).
    Returns (best v on the CPU, record)."""
    if score_fn is None:
        z_clean = features_of(feat_fn, build_u8, device)
        score_fn = lambda x, idx: token_cosine(z_clean[idx.to(z_clean.device)], feat_fn(x))

    def build_score(v):   # all build images, perturbed images rounded to 8 bits as in the evaluation
        out = []
        with torch.no_grad():
            for i in range(0, len(build_u8), eval_bs):
                x = build_u8[i:i + eval_bs].to(device).float() / 255
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    out.append(score_fn(quantize(x + v), torch.arange(i, i + len(x))).float().cpu())
        return torch.cat(out).mean().item()

    v = (torch.zeros(1, 3, build_u8.shape[-2], build_u8.shape[-1]) if v0 is None else v0.detach().cpu().clone())
    v = v.to(device).requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    st = {'epoch': 0, 'lr': lr, 'bad': 0, 'halvings': 0, 'history': []}
    if ckpt_path is not None and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location='cpu')
        with torch.no_grad():
            v.copy_(ck['v'].to(device))
        opt.load_state_dict(ck['opt'])
        st, best_v = ck['state'], ck['best_v']
        print('>> %s: resuming at epoch %d' % (name, st['epoch']))
    else:
        best_v = v.detach().cpu().clone()
        st['best'] = build_score(v.detach())
        st['best_epoch'] = 0
        print('  %s start: score on the build images %.4f' % (name, st['best']))
    t0, stop = time.time(), False
    while st['epoch'] < max_epochs:
        for g in opt.param_groups:
            g['lr'] = st['lr']
        losses = []
        perm = torch.randperm(len(build_u8), generator=torch.Generator().manual_seed(1000 * seed + st['epoch']))
        for idx in perm.split(batch_size):
            x = build_u8[idx].to(device).float() / 255
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = score_fn((x + v).clamp(0, 1), idx).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                v.clamp_(-xi, xi)
            losses.append(loss.item())
        st['epoch'] += 1
        score = build_score(v.detach())
        improved = score < st['best'] - min_delta
        if score < st['best']:
            st['best'], st['best_epoch'], best_v = score, st['epoch'], v.detach().cpu().clone()
        st['bad'] = 0 if improved else st['bad'] + 1
        st['history'].append({'epoch': st['epoch'], 'lr': st['lr'], 'mean_loss': float(np.mean(losses)),
                              'build_score': score, 'minutes': (time.time() - t0) / 60})
        print('  %s epoch %d: loss %.4f during the epoch, %.4f at its end (best %.4f, epoch %d), lr %.2e, %.1f min' % (
            name, st['epoch'], np.mean(losses), score, st['best'], st['best_epoch'], st['lr'], (time.time() - t0) / 60))
        stop = False
        if st['bad'] >= plateau:
            if st['halvings'] >= max_halvings:
                stop = st['epoch'] >= min_epochs
            else:   # plateau: restart from the best v with half the step
                st['halvings'] += 1
                st['lr'] /= 2
                st['bad'] = 0
                with torch.no_grad():
                    v.copy_(best_v.to(device))
                opt = torch.optim.Adam([v], lr=st['lr'])
                print('  %s: plateau, restart from the best v with lr %.2e' % (name, st['lr']))
        if ckpt_path is not None:
            tmp = Path(str(ckpt_path) + '.tmp')
            torch.save({'v': v.detach().cpu(), 'opt': opt.state_dict(), 'state': st, 'best_v': best_v}, tmp)
            tmp.replace(ckpt_path)
        if stop:
            break
    st['converged'] = bool(stop)   # False: stopped by max_epochs before the plateau criterion
    print('  %s: kept epoch %d of %d, score on the build images %.4f%s' % (
        name, st['best_epoch'], st['epoch'], st['best'], '' if st['converged'] else ' (max_epochs reached)'))
    return best_v, st
