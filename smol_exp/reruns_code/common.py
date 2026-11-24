"""Shared code for the re-runs: data, models, layer-wise encoder forward, training, evaluation.

Mirrors the group's notebooks (smol_exp/01 and 02) so numbers stay comparable:
same ImageNetV2 permutation (seed 0), same 500 build / held-out images, same eps = 10/255,
Adam lr 1/255, batch 8, 10 epochs, same caption prompt, greedy decoding, MiniLM threshold 0.5.
"""
import glob, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
PERT = ROOT / "perturbations"; PERT.mkdir(exist_ok=True)
DATA_DIR = Path("C:/Users/david/uap_data")
GROUP_V = ROOT.parent / "repo" / "smol_exp" / "uap_smolvlm" / "uap_smolvlm256m_linf10.npy"

M256 = "HuggingFaceTB/SmolVLM-256M-Instruct"
M500 = "HuggingFaceTB/SmolVLM-500M-Instruct"
M2B = "HuggingFaceTB/SmolVLM-Instruct"
IMG = 512
N_BUILD, N_EVAL = 500, 500
XI = 10 / 255
PROMPT = "Describe this image in one sentence."
MAX_NEW = 40
THR = 0.5
SEED = 0
DEV = "cuda"

try:  # keep Windows from sleeping while a run is in progress (released automatically when the process exits)
    import ctypes
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED: on "modern standby" laptops the display timeout
    # is what triggers standby (which suspends this process), so the display must be kept on too.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
except Exception:
    pass

torch.backends.cuda.matmul.allow_tf32 = False   # keep fp32 evaluation exact, like the group's T4 runs
torch.backends.cudnn.allow_tf32 = False


# ----------------------------------------------------------------------------- data
def files_split():
    # Sort on POSIX-style relative paths so the order matches the group's Linux/Colab glob exactly
    # (on Windows, sorting raw paths with backslashes would order the class folders differently).
    rel = sorted(Path(f).relative_to(DATA_DIR).as_posix() for f in glob.glob(str(DATA_DIR / "imagenetv2-*" / "*" / "*.jpeg")))
    assert len(rel) == 10000, len(rel)
    perm = np.random.RandomState(SEED).permutation(len(rel))
    build = [rel[i] for i in perm[:N_BUILD]]
    ev = [rel[i] for i in perm[N_BUILD:N_BUILD + N_EVAL]]
    return build, ev


def load_u8(which):
    """uint8 tensor (N,3,512,512): Resize(512) + CenterCrop(512), as in the notebooks. Cached on disk."""
    cache = DATA_DIR / f"{which}_u8.pt"
    if cache.exists():
        return torch.load(cache)
    from torchvision import transforms
    pre = transforms.Compose([transforms.Resize(IMG), transforms.CenterCrop(IMG), transforms.PILToTensor()])
    build, ev = files_split()
    files = build if which == "build" else ev
    x = torch.stack([pre(Image.open(DATA_DIR / f).convert("RGB")) for f in files])
    torch.save(x, cache)
    return x


def labels(which):
    build, ev = files_split()
    return [int(f.split("/")[1]) for f in (build if which == "build" else ev)]


def to01(u8): return u8.to(DEV).float().div(255)


def load_v(name):
    p = GROUP_V if name == "group_final" else PERT / f"v_{name}.npy"
    return torch.from_numpy(np.load(p)).float().to(DEV)


def random_v():
    torch.manual_seed(SEED)
    return XI * torch.sign(torch.randn(1, 3, IMG, IMG, device=DEV))


# ----------------------------------------------------------------------------- models
def load_vlm(mid=M256, dtype=torch.float32):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(mid)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return proc, model


def norm(x01):                      # SmolVLM image_mean = image_std = 0.5
    return (x01 - 0.5) / 0.5


def encoder_states(vis, pixel_values, upto=None):
    """Layer-wise forward of the SmolVLM vision encoder.

    Returns a dict: 'h0'..'h12' (residual stream after embeddings / after layer k) and 'final'
    (= post_layernorm(h12) = last_hidden_state, the group's target). If `upto` is an int, stops after
    that layer (cheaper when training on a middle layer)."""
    B = pixel_values.shape[0]
    ps = vis.config.patch_size
    mask = torch.ones(B, pixel_values.shape[2] // ps, pixel_values.shape[3] // ps, dtype=torch.bool, device=pixel_values.device)
    h = vis.embeddings(pixel_values=pixel_values, patch_attention_mask=mask)
    out = {"h0": h}
    n = len(vis.encoder.layers) if upto is None else upto
    for k, layer in enumerate(vis.encoder.layers[:n]):
        h = layer(h, None)
        out[f"h{k + 1}"] = h
    if upto is None:
        out["final"] = vis.post_layernorm(h)
    return out


def tok_cos(a, b):                  # mean over tokens of per-token cosine -> (B,)
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean(dim=1)


# ----------------------------------------------------------------------------- training
TARGETS = {                         # which encoder states the loss looks at
    "final": ["final"],
    "h3": ["h3"], "h6": ["h6"], "h9": ["h9"],
    "multi": ["h3", "h6", "h9", "final"],
    "multi_mid": ["h3", "h6", "h9"],                                  # same as multi without the output
    "multi_all": [f"h{k}" for k in range(1, 13)] + ["final"],          # every layer + output
}


def train_uap(vis, build_u8, target, epochs=10, bs=8, lr=1 / 255, xi=XI, log=print, amp=True):
    keys = TARGETS[target]
    upto = None if "final" in keys else max(int(k[1:]) for k in keys)
    g = torch.Generator().manual_seed(SEED)
    v = torch.zeros(1, 3, IMG, IMG, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr)
    hist, t0, n = [], time.time(), len(build_u8)
    for ep in range(epochs):
        losses = []
        for idx in torch.randperm(n, generator=g).split(bs):
            x = to01(build_u8[idx])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                with torch.no_grad():
                    zc = encoder_states(vis, norm(x), upto)
                za = encoder_states(vis, norm((x + v).clamp(0, 1)), upto)
                loss = torch.stack([tok_cos(zc[k], za[k]).mean() for k in keys]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v.clamp_(-xi, xi)
            losses.append(loss.item())
        hist.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "minutes": (time.time() - t0) / 60})
        log(f"[{target}] epoch {ep + 1}/{epochs}  loss (mean cos on {keys}) = {hist[-1]['loss']:.3f}  ({hist[-1]['minutes']:.1f} min)")
        np.save(PERT / f"v_{target}.npy", v.detach().cpu().numpy())
    return v.detach(), hist


# ----------------------------------------------------------------------------- evaluation helpers
def chat(proc, prompt=PROMPT):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True)


def to_pil(u8, pert=None):
    """Real pipeline: add v in pixel space, round to 8-bit, as an attacker saving an image file."""
    x = u8.float()
    if pert is not None:
        x = x + pert[0].cpu() * 255
    return Image.fromarray(x.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy())


@torch.no_grad()
def captions(proc, model, u8s, pert, split=False, bs=8, prompt=PROMPT):
    outs = []
    dtype = next(model.parameters()).dtype
    for i in range(0, len(u8s), bs):
        pils = [to_pil(u, pert) for u in u8s[i:i + bs]]
        inp = proc(text=[chat(proc, prompt)] * len(pils), images=[[p] for p in pils],
                   do_image_splitting=split, return_tensors="pt", padding=True).to(DEV)
        inp["pixel_values"] = inp["pixel_values"].to(dtype)
        g = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
        outs += proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]


@torch.no_grad()
def layer_damage(proc, vis, u8s, pert, bs=10):
    """Real pipeline, splitting off: mean token cosine clean vs perturbed at every encoder state."""
    acc = {}
    for i in range(0, len(u8s), bs):
        pv = []
        for p in (None, pert):
            px = proc.image_processor(images=[[to_pil(u, p)] for u in u8s[i:i + bs]], do_image_splitting=False,
                                      return_tensors="pt")["pixel_values"]
            pv.append(px.flatten(0, 1).to(DEV, next(vis.parameters()).dtype))
        s0, s1 = encoder_states(vis, pv[0]), encoder_states(vis, pv[1])
        for k in s0:
            acc.setdefault(k, []).append(tok_cos(s0[k], s1[k]).cpu())
    return {k: torch.cat(v).mean().item() for k, v in acc.items()}


_ST = None
def text_sim(a, b):
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)
    ea = _ST.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    eb = _ST.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    return (ea * eb).sum(-1).cpu()


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def changed_stats(clean, pert):
    sim = text_sim(clean, pert)
    k5, k7, n = int((sim < 0.5).sum()), int((sim < 0.7).sum()), len(sim)
    return {"n": n, "changed": k5 / n, "ci95": wilson(k5, n), "changed@0.7": k7 / n, "mean_sim": sim.mean().item()}


def save_json(name, obj):
    p = RES / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)
    return p


def load_json(name, default=None):
    p = RES / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else default


class Logger:
    def __init__(self, name):
        self.f = open(RES / name, "a", encoding="utf-8")
    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); self.f.write(s + "\n"); self.f.flush()
