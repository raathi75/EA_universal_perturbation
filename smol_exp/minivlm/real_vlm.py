"""Universal perturbation on REAL, pre-trained VLMs (trained by their authors on their own data, encoder and LM
different from the mini-VLMs), with exactly the protocol of the mini-VLMs:

  attack:      label-free universal v on the vision encoder tokens (the tokens the projector receives), built on the
               500 ImageNetV2 build images and run to convergence (common.converge_uap), l_inf 10/255
  evaluation:  1,000 COCO val images (CIDEr, wrong vs the 5 human captions, changed) and the 500 ImageNetV2
               held-out images (changed @0.5 and @0.7), for: clean, own v, random noise, and as transfers the
               perturbations of the mini-VLMs, of SmolVLM (report) and of the other real VLMs
  real pipeline: the captions are generated through the model's OWN processor from 8-bit PIL images (perturbed
               images rounded to 8 bits); the gradient for the attack goes through a differentiable copy of the
               preprocessing, checked against the processor at start-up (sanity check)

One fixed square resolution per model, one view (no dynamic tiling, as the report's "splitting off"); images are
Resize + CenterCrop, the same field of view as the 224 crops of the mini-VLMs.

Models (native in transformers, no remote code):
  smolvlm-256m   HuggingFaceTB/SmolVLM-256M-Instruct        SigLIP-B/16 (512)  + SmolLM2-135M   (the report's model)
  llava-ov-0.5b  llava-hf/llava-onevision-qwen2-0.5b-ov-hf  SigLIP-SO400M (384) + Qwen2-0.5B
  qwen2-vl-2b    Qwen/Qwen2-VL-2B-Instruct                  Qwen2-VL ViT (448)  + Qwen2-1.5B
  qwen2-vl-2b-224  the same model at 224 x 224 (resolution of the mini-VLMs)
Results in results/<model>/ with the same files and keys as the mini-VLMs (compare.py adds them).
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from common import (CAPTION_SIM_CHANGED, DATA_DIR, ENCODERS, HERE, MAX_NEW_TOKENS, PROMPT, SEED, XI, CocoCaptions,
                    TextSim, converge_uap, extra_conditions, seed_start, seed_tag, feature_cosine, features_of, get_device, imagenetv2_split, results_dir, token_cosine)
from eval_coco import rate, tokenize
from evaluate_captions import rates

SPECS = {
    'smolvlm-256m': dict(hf='HuggingFaceTB/SmolVLM-256M-Instruct', size=512, kind='smolvlm'),
    'llava-ov-0.5b': dict(hf='llava-hf/llava-onevision-qwen2-0.5b-ov-hf', size=384, kind='llava_ov'),
    'qwen2-vl-2b': dict(hf='Qwen/Qwen2-VL-2B-Instruct', size=448, kind='qwen2_vl'),
    # same model at the resolution of the mini-VLMs (Qwen2-VL takes any multiple of 28): 16 x 16 = 256 patches,
    # 64 tokens for the language model. Tests whether its robustness comes from the resolution
    'qwen2-vl-2b-224': dict(hf='Qwen/Qwen2-VL-2B-Instruct', size=224, kind='qwen2_vl'),
}


def load_square(paths, size):
    pre = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.PILToTensor()])
    return torch.stack([pre(Image.open(p).convert('RGB')) for p in paths])


def imagenetv2_at(size):
    cache = DATA_DIR / ('imagenetv2_split_%d.pt' % size)
    if cache.exists():
        d = torch.load(cache)
        return d['build'], d['eval']
    build_files, eval_files = imagenetv2_split()
    d = {'build': load_square(build_files, size), 'eval': load_square(eval_files, size)}
    torch.save(d, cache)
    return d['build'], d['eval']


def resize_v(v, size, xi):
    v = v.float().reshape(1, 3, v.shape[-2], v.shape[-1])
    if v.shape[-1] != size:
        v = F.interpolate(v, size=size, mode='bilinear', antialias=True, align_corners=False)
    return v.clamp(-xi, xi)


def load_any_v(path, size, xi, device):
    path = str(path)
    v = torch.from_numpy(np.load(path)) if path.endswith('.npy') else torch.load(path, map_location='cpu')
    return resize_v(v, size, xi).to(device)


def u8_to_pil(u8):
    return Image.fromarray(u8.permute(1, 2, 0).cpu().numpy())


class RealVLM:
    def __init__(self, name, device):
        import transformers
        self.name, self.device = name, device
        spec = SPECS[name]
        self.size, self.kind = spec['size'], spec['kind']
        self.processor = transformers.AutoProcessor.from_pretrained(spec['hf'])
        tok = getattr(self.processor, 'tokenizer', None)
        if tok is not None:
            tok.padding_side = 'left'
        cls = {'smolvlm': 'AutoModelForImageTextToText', 'llava_ov': 'LlavaOnevisionForConditionalGeneration',
               'qwen2_vl': 'Qwen2VLForConditionalGeneration'}[self.kind]
        kw = {'attn_implementation': 'sdpa'} if self.kind == 'qwen2_vl' else {}
        self.model = getattr(transformers, cls).from_pretrained(spec['hf'], torch_dtype=torch.bfloat16, **kw).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        ip = self.processor.image_processor
        if self.kind == 'smolvlm':
            # one 512 x 512 view per image, no tiles (recent transformers ignore do_image_splitting as a call argument)
            ip.do_image_splitting = False
            # the processor first resizes to longest_edge 2048 then back to 512: an up/down resampling that changes the
            # image by a few levels. With 512 the 512 x 512 image goes through unchanged, as in the report's notebooks.
            ip.size = {'longest_edge': self.size}
        self.mean = torch.tensor(ip.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=device).view(1, 3, 1, 1)
        m = self.model
        if self.kind == 'smolvlm':
            self.vision = m.model.vision_model
        elif self.kind == 'llava_ov':
            self.vision = getattr(m, 'vision_tower', None) or m.model.vision_tower
            self.layer = m.config.vision_feature_layer
        else:
            self.vision = getattr(m, 'visual', None) or m.model.visual
            self.merge = ip.merge_size
            self.patch = ip.patch_size
            self.tps = ip.temporal_patch_size
            self._captured = None
            # the encoder tokens are the INPUT of the patch merger (which is Qwen2-VL's projector)
            self.vision.merger.register_forward_hook(lambda mod, inp, out: setattr(self, '_captured', inp[0]))
            # activations of 32 blocks x 1024 tokens per image do not fit on 16 GB for a batch of 8: recompute each
            # block in the backward pass (checkpointing, same gradients)
            from torch.utils.checkpoint import checkpoint
            for blk in self.vision.blocks:
                f = blk.forward
                blk.forward = lambda *a, _f=f, **k: (checkpoint(_f, *a, use_reentrant=False, **k)
                                                     if torch.is_grad_enabled() else _f(*a, **k))
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
        self.prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    # ------------------------------------------------------------------ differentiable path (attack)
    def qwen_patches(self, xn):
        """Normalised images B x 3 x S x S -> Qwen2-VL flattened patches, as Qwen2VLImageProcessor does."""
        b, c, h, w = xn.shape
        gh, gw, m, p, t = h // self.patch, w // self.patch, self.merge, self.patch, self.tps
        x = xn.unsqueeze(1).expand(b, t, c, h, w)
        x = x.reshape(b, t, c, gh // m, m, p, gw // m, m, p).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = x.reshape(b * gh * gw, c * t * p * p)
        grid = torch.tensor([[1, gh, gw]] * b, device=xn.device)
        return flat, grid

    def tokens(self, x01):
        """[0, 1] images B x 3 x S x S -> encoder tokens B x N x D (the tokens the projector receives)."""
        xn = (x01 - self.mean) / self.std
        dt = next(self.vision.parameters()).dtype
        if self.kind == 'smolvlm':
            return self.vision(pixel_values=xn.to(dt)).last_hidden_state
        if self.kind == 'llava_ov':
            hs = self.vision(pixel_values=xn.to(dt), output_hidden_states=True).hidden_states
            layers = self.layer if isinstance(self.layer, (list, tuple)) else [self.layer]
            return torch.cat([hs[l] for l in layers], dim=-1)
        out = []
        for i in range(len(xn)):   # one image at a time: the vision attention of Qwen2-VL is quadratic in all tokens
            flat, grid = self.qwen_patches(xn[i:i + 1])
            self.vision(flat.to(dt), grid_thw=grid)
            out.append(self._captured)
        return torch.stack(out)

    # ------------------------------------------------------------------ real pipeline (captions)
    def inputs(self, pils):
        if self.kind == 'smolvlm':
            return self.processor(text=[self.prompt] * len(pils), images=[[p] for p in pils],
                                  return_tensors='pt', padding=True)
        return self.processor(text=[self.prompt] * len(pils), images=pils, return_tensors='pt', padding=True)

    @torch.no_grad()
    def sanity_check(self, u8):
        """The differentiable preprocessing must match the model's processor (on a real image)."""
        x = u8[:1].to(self.device).float() / 255
        mine = (x - self.mean) / self.std
        ref = self.inputs([u8_to_pil(u8[0])])
        pv = ref['pixel_values'].to(self.device).float()
        if self.kind == 'qwen2_vl':
            flat, grid = self.qwen_patches(mine)
            assert torch.equal(grid.cpu(), ref['image_grid_thw'].cpu()), (grid, ref['image_grid_thw'])
            diff = (flat - pv).abs().max().item()
        else:
            # LLaVA-OneVision (anyres) gives a 384 x 384 image twice: the base view and one high-resolution tile,
            # both equal to the image itself. Every view must match, so attacking one view attacks them all.
            pv = pv.reshape(-1, 3, self.size, self.size)
            views = pv.shape[0]
            print('>> processor views per image:', views)
            assert views == 1 or self.kind == 'llava_ov', 'the processor split the image into %d views' % views
            diff = (mine - pv).abs().max().item()
            print('>> mean |differentiable preprocessing - processor| = %.5f' % (mine - pv).abs().mean().item())
        print('>> sanity check: max |differentiable preprocessing - processor| = %.4f (expected < 0.05)' % diff)
        assert diff < 0.05, 'the differentiable preprocessing does not match the processor: send this output'
        t = self.tokens(x)
        print('>> encoder tokens per image:', tuple(t.shape[1:]))

    # ------------------------------------------------------------------ end-to-end attack (caption likelihood)
    @torch.no_grad()
    def caption_ids(self, images_u8, bs=16):
        """Token ids of the clean greedy captions (up to and including the end-of-turn token)."""
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        tok = self.processor.tokenizer
        if tok.pad_token_id is not None:
            eos.add(tok.pad_token_id)
        out = []
        for i in range(0, len(images_u8), bs):
            inp = self.inputs([u8_to_pil(im) for im in images_u8[i:i + bs]]).to(self.device)
            inp = {k: (t.to(torch.bfloat16) if t.is_floating_point() else t) for k, t in inp.items()}
            gen = self.model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            for g in gen[:, inp['input_ids'].shape[1]:].tolist():
                ids = []
                for t in g:
                    ids.append(t)
                    if t in eos:
                        break
                out.append(ids)
        return out

    def setup_caption_attack(self, build_u8, cache):
        """Prompt of one image (identical for every image: same size, same prompt) and clean caption ids."""
        p = self.inputs([u8_to_pil(build_u8[0])])
        self.p_ids = p['input_ids'][:1].to(self.device)
        self.p_extra = {k: t[:1].to(self.device) for k, t in p.items()
                        if k not in ('input_ids', 'attention_mask', 'pixel_values')}
        self.n_views = p['pixel_values'].shape[1] if self.kind != 'qwen2_vl' else 1
        if cache.exists():
            self.cap_ids = json.load(open(cache))
        else:
            print('>> clean captions of the %d build images (target of the end-to-end attack)' % len(build_u8))
            self.cap_ids = self.caption_ids(build_u8)
            json.dump(self.cap_ids, open(cache, 'w'))
        assert len(self.cap_ids) == len(build_u8)

    def caption_score(self, x01, idx):
        """Per-image likelihood of its clean caption, exp(-mean cross-entropy) in [0, 1] (1 = unchanged),
        differentiable w.r.t. the [0, 1] images: lower = the caption is being destroyed."""
        b = len(x01)
        xn = ((x01 - self.mean) / self.std).to(next(self.vision.parameters()).dtype)
        extra = {k: t.repeat(b, *[1] * (t.dim() - 1)) for k, t in self.p_extra.items()}
        if self.kind == 'qwen2_vl':
            pv, extra['image_grid_thw'] = self.qwen_patches(xn)
        elif self.kind == 'llava_ov':
            pv = xn.unsqueeze(1).repeat(1, self.n_views, 1, 1, 1)   # every view is the image itself (sanity check)
        else:
            pv = xn.unsqueeze(1)
        caps = [self.cap_ids[i] for i in idx.tolist()]
        lt, lp = max(len(c) for c in caps), self.p_ids.shape[1]
        pad = self.processor.tokenizer.pad_token_id or 0
        ids = torch.full((b, lt), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((b, lt), dtype=torch.long, device=self.device)
        for i, c in enumerate(caps):
            ids[i, :len(c)] = torch.tensor(c, device=self.device)
            mask[i, :len(c)] = 1
        for k, t in list(extra.items()):   # per-token inputs of the prompt (e.g. mm_token_type_ids): text after it
            if t.dim() == 2 and t.shape[1] == lp and k != 'image_grid_thw':
                extra[k] = torch.cat([t, torch.zeros((b, lt), dtype=t.dtype, device=t.device)], 1)
        input_ids = torch.cat([self.p_ids.expand(b, -1), ids], 1)
        attention = torch.cat([torch.ones((b, lp), dtype=torch.long, device=self.device), mask], 1)
        hidden = self.model.model(input_ids=input_ids, attention_mask=attention, pixel_values=pv, use_cache=False,
                                  **extra).last_hidden_state
        logits = self.model.lm_head(hidden[:, lp - 1:lp - 1 + lt]).float()   # only the caption positions
        ce = F.cross_entropy(logits.transpose(1, 2), ids, reduction='none')
        return torch.exp(-(ce * mask).sum(1) / mask.sum(1))

    @torch.no_grad()
    def caption(self, images_u8, v=None, bs=16):
        outs = []
        for i in range(0, len(images_u8), bs):
            u8 = images_u8[i:i + bs]
            if v is not None:
                u8 = torch.round(((u8.to(self.device).float() / 255 + v).clamp(0, 1)) * 255).to(torch.uint8)
            inp = self.inputs([u8_to_pil(im) for im in u8]).to(self.device)
            inp = {k: (t.to(torch.bfloat16) if t.is_floating_point() else t) for k, t in inp.items()}
            gen = self.model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            outs += [o.strip() for o in self.processor.batch_decode(gen[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)]
        return outs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universal perturbation on a real pre-trained VLM')
    parser.add_argument('--model', required=True, choices=list(SPECS))
    parser.add_argument('--n_coco', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16, help='captioning batch size')
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--target', default='tokens', choices=['tokens', 'caption'],
                        help='tokens: label-free, encoder tokens (as in the report); caption: end-to-end, likelihood of '
                             'the clean captions of the build images (the strongest test of robustness)')
    parser.add_argument('--lr', type=float, default=1, help='initial Adam step on the [0, 255] scale (halved on plateaus)')
    parser.add_argument('--attack_batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0,
                        help='0 = the original run; k > 0: random start in the ball and another image order -> *_s<k>.pt')
    args = parser.parse_args()

    device = get_device()
    out = results_dir(args.model)
    vlm = RealVLM(args.model, device)
    S = vlm.size
    build_u8, eval_u8 = imagenetv2_at(S)
    vlm.sanity_check(eval_u8)

    # ------------------------------------------------------------------ 1. own universal v, to convergence
    file_v, file_cap = out / 'uap_xi10.pt', out / 'uap_cap_xi10.pt'
    tag = ('uap_xi10' if args.target == 'tokens' else 'uap_cap_xi10') + seed_tag(args.seed)
    if not (out / (tag + '.pt')).exists():
        torch.manual_seed(SEED + args.seed)
        kw = {}
        if args.target == 'caption':
            vlm.setup_caption_attack(build_u8, out / 'build_caption_ids.json')
            kw = dict(score_fn=vlm.caption_score, eval_bs=args.attack_batch_size)
        v, rec = converge_uap(vlm.tokens, build_u8, device, v0=seed_start(args.seed, S, XI), seed=args.seed,
                              xi=XI, lr=args.lr / 255, batch_size=args.attack_batch_size,
                              ckpt_path=out / (tag + '_ckpt.pt'), max_epochs=args.max_epochs,
                              name='%s %s' % (args.model, args.target), **kw)
        torch.save(v, out / (tag + '.pt'))
        json.dump(rec, open(out / (tag + '_history.json'), 'w'), indent=1)
        (out / (tag + '_ckpt.pt')).unlink(missing_ok=True)

    torch.manual_seed(SEED)
    conds = {'clean': None}
    if file_v.exists():
        conds['universal v (%s)' % args.model] = load_any_v(file_v, S, XI, device)
    if file_cap.exists():
        conds['universal v (%s, caption loss)' % args.model] = load_any_v(file_cap, S, XI, device)
    conds['random ±xi'] = XI * torch.sign(torch.randn(1, 3, S, S, device=device))
    sources = {e: results_dir(e) / 'uap_xi10.pt' for e in ENCODERS}           # mini-VLMs (224)
    sources['smolvlm (report)'] = HERE.parent / 'uap_smolvlm' / 'uap_smolvlm256m_linf10.npy'
    sources.update({m: results_dir(m) / 'uap_xi10.pt' for m in SPECS if m != args.model})
    for src, p in sources.items():
        if p.exists():
            conds['universal v (%s), transfer' % src] = load_any_v(p, S, XI, device)
        else:
            print('>> Transfer skipped (not computed yet):', src)
    for name, p in extra_conditions(out, 10, S, device, 'universal v (%s)' % args.model).items():
        conds[name] = p if torch.is_tensor(p) else load_any_v(p, S, XI, device)

    # ------------------------------------------------------------------ 2. token damage (held-out ImageNetV2)
    tok = {'model': args.model, 'size': S, 'token_cos': {}}
    clean_t = features_of(vlm.tokens, eval_u8, device, bs=8)
    for name, v in conds.items():
        if v is not None:
            tok['token_cos'][name] = feature_cosine(vlm.tokens, eval_u8, device, clean_t, v, bs=8).mean().item()
            print('%-55s token cos %.3f' % (name, tok['token_cos'][name]))
    json.dump(tok, open(out / 'uap_xi10_tokens.json', 'w'), indent=1)
    del clean_t

    # ------------------------------------------------------------------ 3. captions: COCO val and ImageNetV2 held-out
    from pycocoevalcap.cider.cider import Cider
    sim_fn = TextSim(device)
    coco = CocoCaptions('val2017', max_images=args.n_coco)
    coco_u8 = load_square([p for p, _ in coco.items], S)
    refs = [r for _, r in coco.items]
    gts = {i: [tokenize(c) for c in r] for i, r in enumerate(refs)}
    ref_emb = sim_fn.st.encode([c for r in refs for c in r], convert_to_tensor=True, normalize_embeddings=True)
    owner = torch.tensor([i for i, r in enumerate(refs) for _ in r], device=ref_emb.device)

    f_c, f_cc = out / 'coco_stage2.json', out / 'coco_captions_stage2.json'
    f_i, f_ic = out / 'eval_captions_stage2_xi10.json', out / 'captions_stage2_xi10.json'
    res_c = json.load(open(f_c)) if f_c.exists() else {'model': args.model, 'results': {}}
    caps_c = json.load(open(f_cc)) if f_cc.exists() else {}
    res_i = json.load(open(f_i)) if f_i.exists() else {'model': args.model, 'results': {}}
    caps_i = json.load(open(f_ic)) if f_ic.exists() else {}
    res_c['n_images'], res_i['n_images'], res_c['size'] = len(coco_u8), len(eval_u8), S

    for name, v in conds.items():
        if name not in res_c['results']:
            print('>> COCO val, %d images: %s' % (len(coco_u8), name))
            caps_c[name] = vlm.caption(coco_u8, v, args.batch_size)
            cider, _ = Cider().compute_score(gts, {i: [tokenize(c)] for i, c in enumerate(caps_c[name])})
            e = sim_fn.st.encode(caps_c[name], convert_to_tensor=True, normalize_embeddings=True)
            s = (e[owner] * ref_emb).sum(-1)
            best = torch.full((len(e),), -1.0, device=s.device).scatter_reduce(0, owner, s, reduce='amax').cpu()
            r = {'CIDEr': cider, 'mean_best_human_sim': best.mean().item(), 'wrong': rate(best < CAPTION_SIM_CHANGED)}
            if v is not None:
                r['changed'] = rate(sim_fn(caps_c['clean'], caps_c[name]) < CAPTION_SIM_CHANGED)
            res_c['results'][name] = r
            json.dump(caps_c, open(f_cc, 'w'), indent=1)
            json.dump(res_c, open(f_c, 'w'), indent=1)
        if name not in res_i['results'] and (v is not None or 'clean' not in caps_i):
            print('>> ImageNetV2 held-out, %d images: %s' % (len(eval_u8), name))
            caps_i[name] = vlm.caption(eval_u8, v, args.batch_size)
            if v is not None:
                sim = sim_fn(caps_i['clean'], caps_i[name])
                res_i['results'][name] = {'mean_caption_sim': sim.mean().item(),
                                          'changed@0.5': rates(sim, CAPTION_SIM_CHANGED), 'changed@0.7': rates(sim, 0.7)}
            json.dump(caps_i, open(f_ic, 'w'), indent=1)
            json.dump(res_i, open(f_i, 'w'), indent=1)

    print('\n%s (%d px), COCO val (%d images) and ImageNetV2 held-out (%d images), xi = 10/255' % (
        args.model, S, len(coco_u8), len(eval_u8)))
    print('%-55s %7s %8s %9s   %s' % ('condition', 'CIDEr', 'wrong', 'changed', 'ImageNetV2 changed@0.5'))
    for name, r in res_c['results'].items():
        i = res_i['results'].get(name, {}).get('changed@0.5')
        print('%-55s %7.3f %7.1f%% %9s   %s' % (
            name, r['CIDEr'], r['wrong']['rate'], '%.1f%%' % r['changed']['rate'] if 'changed' in r else '—',
            '%.1f%% (%.1f-%.1f)' % (i['all']['rate'], *i['all']['ci95']) if i else '—'))
    own = 'universal v (%s)' % args.model
    print('\nExamples (human / clean / own v):')
    for k in range(3):
        print('  human:', refs[k][0], '\n  clean:', caps_c['clean'][k], '\n  v    :', caps_c.get(own, ['—'] * 3)[k], '\n')
    print('>> Saved', out)
