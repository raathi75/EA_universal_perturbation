# ================================================================================================
# Re-evaluation on Colab (T4 GPU): per-image data the notebooks never saved.
#
# Nothing is trained. Every perturbation is loaded from Drive (uap_smolvlm/). Nothing existing is
# overwritten: outputs go to a NEW folder, /content/drive/MyDrive/uap_smolvlm/stats_audit_reeval/.
# Each output is a CSV with one row per image, keyed by image_id (the ImageNetV2 file path, the
# same key as stats_audit/data/heldout_files.json). Copy that folder to stats_audit/reeval/ and
# rerun audit.py.
#
# The generation code is copied from the notebooks (same processor calls, batch sizes, seeds,
# noise draws), and every part prints a check against the aggregate the notebook saved.
#
# Usage: Runtime -> T4 GPU, paste this file in one cell (or upload it and `%run reeval_colab.py`).
# Choose parts in RUN below. Rough T4 times: A 3 min, B 8 min, C 5 min, D 10 min, E 25 min, F 20 min.
# ================================================================================================
RUN = {
    "A_tokcos_after_preproc": True,   # §5 token cosines (500 imgs): v, random (CUDA draw), GoogLeNet v  -> SD
    "B_tokcos_psnr_pipeline": True,   # §6/§9 token cosine + PSNR, real pipeline off (50 imgs), 4 budgets -> SD
    "C_tiles_split_on": True,         # §6/§7 per-tile token cosines, splitting on (50 imgs)            -> SD
    "D_prompts_nb02": True,           # §5 prompt table + yes/no breakdown (100 imgs), B1/B2 random (CUDA)
    "E_defenses_nb05": True,          # §11.1 JPEG, blur (200), global tile blanked (100): captions
    "F_transfer_nb03": True,          # §8 500M (200) and 2.2B (100): captions + token cosine (30)
}

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers>=4.50", "sentence-transformers"], check=True)
import io, glob, tarfile, csv, time, gc
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms
from tqdm.auto import tqdm
from google.colab import drive
drive.mount("/content/drive")
DEVICE = "cuda"; assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"

IMG_SIZE, N_BUILD, SEED, MAX_NEW_TOKENS = 512, 500, 0, 40
XI = 10 / 255
VDIR = Path("/content/drive/MyDrive/uap_smolvlm")
REPL = Path("/content/drive/MyDrive/uap_replication/universal_googlenet_linf.npy")
OUT = VDIR / "stats_audit_reeval"; OUT.mkdir(parents=True, exist_ok=True)

# Input files this script reads (copy them from the repository's smol_exp/uap_smolvlm/ folder into VDIR).
NEEDED = {f"uap_smolvlm256m_linf{e}.npy": "A-F" for e in (4, 8, 10, 16)}
NEEDED.update({"adaptive_v10_JPEG_q75.npy": "E", "adaptive_v10_blur_σ1.npy": "E",
               "adaptive_v10_global_tile__blanked.npy": "E"})
missing = [f"{name} (part {parts})" for name, parts in NEEDED.items() if not (VDIR / name).exists()]
if missing:
    raise SystemExit("Missing in " + str(VDIR) + ":\n  " + "\n  ".join(missing)
                     + "\nUpload them from smol_exp/uap_smolvlm/ in the repository, then run again.")
if not REPL.exists():
    print(f"Note: {REPL} not found. Part A will skip the GoogLeNet row "
          "(upload 'UAP Notebooks/CNN Experiment utils/universal_googlenet_linf.npy' there to include it).")
P_DESCRIBE = "Describe this image in one sentence."
PROMPTS = {"describe (training)": "Describe this image in one sentence.",
           "what is in it": "What is in this image?",
           "main object": "What is the main object in this picture? Answer in a few words.",
           "animal yes/no": "Is there an animal in this image? Answer yes or no."}

# ---------------- data: identical to every notebook ----------------
DATA_DIR = Path("/content/data"); DATA_DIR.mkdir(exist_ok=True)
TAR = DATA_DIR / "imagenetv2-matched-frequency.tar.gz"
if not list(DATA_DIR.glob("imagenetv2-*/")):
    subprocess.run(["wget", "-q", "-O", str(TAR),
                    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"], check=True)
    with tarfile.open(TAR, "r:*") as t: t.extractall(DATA_DIR)
FILES = sorted(glob.glob(str(DATA_DIR / "imagenetv2-*" / "*" / "*.jpeg")))
assert len(FILES) == 10000, len(FILES)
perm = np.random.RandomState(SEED).permutation(len(FILES))
files_eval = [FILES[i] for i in perm[N_BUILD:N_BUILD + 200]]
IDS = [str(Path(f).relative_to(DATA_DIR)).replace("\\", "/") for f in files_eval]     # image_id
PREPROC = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE), transforms.PILToTensor()])
E_u8 = torch.stack([PREPROC(Image.open(f).convert("RGB")) for f in tqdm(files_eval, desc="decoding")])
print("first image_id:", IDS[0], "(expect .../17/e98b0597...jpeg)")

def write_csv(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("  saved", OUT / name)

from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
def text_sim(a, b):
    ea = st.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    eb = st.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    return (ea * eb).sum(-1).cpu()
def changed(a, b): return 100 * (text_sim(a, b) < 0.5).float().mean().item()
def check(label, got, saved): print(f"  CHECK {label}: recomputed {got:.3f}  saved {saved:.3f}  {'OK' if abs(got - saved) < 0.006 else '<-- DIFFERENT'}")

# ---------------- model (256M) and helpers copied from the notebooks ----------------
from transformers import AutoProcessor, AutoModelForImageTextToText
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID); processor.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(DEVICE).eval()
for p in model.parameters(): p.requires_grad_(False)
VIS = model.model.vision_model; ip = processor.image_processor
MEAN = torch.tensor(ip.image_mean, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(ip.image_std, device=DEVICE).view(1, 3, 1, 1)
def to01(u8): return u8.to(DEVICE).float().div(255)
def norm(x01): return (x01 - MEAN) / STD
def visual_tokens(x01): return VIS(pixel_values=norm(x01)).last_hidden_state
def token_cosine(a, b): return F.cosine_similarity(a, b, dim=-1).mean(dim=1)
def chat(prompt, proc=processor):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True)
def to_pil_u8(u8, pert=None):
    x = u8.float()
    if pert is not None: x = x + pert[0].cpu() * 255
    return Image.fromarray(x.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy())

@torch.no_grad()
def gen_tensor(x01, prompt, bs=16):                       # nb01 caption() / nb02 gen_tensor(): after preprocessing
    dummy = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    base = processor(text=chat(prompt), images=[dummy], do_image_splitting=False, return_tensors="pt")
    outs = []
    for i in range(0, len(x01), bs):
        xb = x01[i:i+bs]; B = len(xb)
        inp = {k: t.repeat(B, *[1] * (t.dim() - 1)).to(DEVICE) for k, t in base.items() if k != "pixel_values"}
        inp["pixel_values"] = norm(xb).unsqueeze(1)
        g = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        outs += processor.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]

@torch.no_grad()
def gen_pipeline(u8s, prompt, pert, split, bs=8, proc=processor, mdl=model):   # nb02/nb03 gen_pipeline
    dtype = next(mdl.parameters()).dtype; outs = []
    for i in range(0, len(u8s), bs):
        pils = [to_pil_u8(u, pert) for u in u8s[i:i+bs]]
        inp = proc(text=[chat(prompt, proc)] * len(pils), images=[[p] for p in pils],
                   do_image_splitting=split, return_tensors="pt", padding=True)
        inp = {k: (t.to(DEVICE, dtype) if t.is_floating_point() else t.to(DEVICE)) for k, t in inp.items()}
        g = mdl.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        outs += proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]

@torch.no_grad()
def tile_cosines(u8s, pert, split, proc=processor, mdl=model):            # nb03 tile_cosines: (n_images, n_tiles)
    dtype = next(mdl.parameters()).dtype; V = mdl.model.vision_model; rows = []
    for u in u8s:
        z = []
        for p in (None, pert):
            px = proc.image_processor(images=[[to_pil_u8(u, p)]], do_image_splitting=split, return_tensors="pt")["pixel_values"][0]
            z.append(V(pixel_values=px.to(DEVICE, dtype)).last_hidden_state.float())
        rows.append(F.cosine_similarity(z[0], z[1], dim=-1).mean(dim=1).cpu())
    return torch.stack(rows)

def psnr_one(u, pert):                                                     # nb04 psnr, per image
    a = u.float(); b = torch.from_numpy(np.asarray(to_pil_u8(u, pert))).permute(2, 0, 1).float()
    return 10 * np.log10(255 ** 2 / max(((a - b) ** 2).mean().item(), 1e-10))

def load_v(name): return torch.from_numpy(np.load(VDIR / name).copy()).float()      # CPU, as in nb03-nb05
V10 = load_v("uap_smolvlm256m_linf10.npy")
V = {e: (V10 if e == 10 else load_v(f"uap_smolvlm256m_linf{e}.npy")) for e in (4, 8, 10, 16)}
# Two different random draws were used in the project (same seed, different device):
torch.manual_seed(SEED); RAND_CUDA = XI * torch.sign(torch.randn_like(V10.to(DEVICE)))   # nb01, nb02 (v on GPU)
torch.manual_seed(SEED); RAND_CPU_UNIT = torch.sign(torch.randn_like(V10))               # nb03, nb04, nb05 (v on CPU)
RAND_CPU = {e: (e / 255) * RAND_CPU_UNIT for e in (4, 8, 10, 16)}


# ================= A. §5 token cosines, after preprocessing, 500 held-out images =================
if RUN["A_tokcos_after_preproc"]:
    print("\nA. token cosine after preprocessing (500 images)")
    files500 = [FILES[i] for i in perm[N_BUILD:N_BUILD + 500]]
    ids500 = [str(Path(f).relative_to(DATA_DIR)).replace("\\", "/") for f in files500]
    E500 = torch.stack([PREPROC(Image.open(f).convert("RGB")) for f in tqdm(files500, desc="decoding 500")])
    perts = {"universal v": V10.to(DEVICE), "random (CUDA draw)": RAND_CUDA}
    if REPL.exists():
        vg = torch.from_numpy(np.load(REPL)).to(DEVICE).float()
        perts["GoogLeNet v"] = F.interpolate(vg, size=IMG_SIZE, mode="bilinear", align_corners=False).clamp(-XI, XI)
    saved = {"universal v": -0.05100833624601364, "random (CUDA draw)": 0.8455979228019714, "GoogLeNet v": 0.8497748374938965}
    rows = []
    with torch.no_grad():
        for name, pert in perts.items():
            cs = []
            for i in range(0, 500, 16):
                x = to01(E500[i:i+16]); cs.append(token_cosine(visual_tokens(x), visual_tokens((x + pert).clamp(0, 1))).cpu())
            cs = torch.cat(cs).numpy()
            check(f"§5 mean token cos, {name}", float(cs.mean()), saved[name])
            rows += [{"condition": name, "image_id": i_, "pos": k, "token_cos": float(c)} for k, (i_, c) in enumerate(zip(ids500, cs))]
    write_csv("A_tokcos_after_preproc.csv", rows)
    del E500; gc.collect()


# ================= B. §6/§9 token cosine + PSNR, real pipeline off, 50 images =================
if RUN["B_tokcos_psnr_pipeline"]:
    print("\nB. token cosine (real pipeline, splitting off) and PSNR, 50 images")
    saved = {4: (0.6228094381093979, 37.04588974970659), 8: (0.29121662557125094, 31.97694873440857),
             10: (0.14212781488895415, 30.075100815216132), 16: (-0.0540974708867725, 26.426038750209223)}
    rows = []
    for e in (4, 8, 10, 16):
        for name, pert in [("universal v", V[e]), ("random (CPU draw)", RAND_CPU[e])]:
            cos = tile_cosines(E_u8[:50], pert, False)[:, 0].numpy()
            ps = np.array([psnr_one(u, pert) for u in E_u8[:50]])
            if name == "universal v":
                check(f"§9 token cos {e}/255", float(cos.mean()), saved[e][0]); check(f"§9 PSNR {e}/255", float(ps.mean()), saved[e][1])
            rows += [{"eps": e, "condition": name, "image_id": IDS[k], "pos": k, "token_cos": float(cos[k]), "psnr_db": float(ps[k])} for k in range(50)]
    write_csv("B_tokcos_psnr_pipeline.csv", rows)


# ================= C. §6/§7 per-tile cosines, splitting on, 50 images =================
if RUN["C_tiles_split_on"]:
    print("\nC. per-tile token cosine, splitting on, 50 images (last tile = global)")
    rows = []
    for name, pert, saved_local, saved_global in [("universal v", V10, 0.723, 0.142), ("random (CPU draw)", RAND_CPU[10], 0.771, 0.865)]:
        T = tile_cosines(E_u8[:50], pert, True).numpy()                   # (50, 17)
        check(f"§7 local tiles, {name}", float(T[:, :-1].mean()), saved_local); check(f"§7 global tile, {name}", float(T[:, -1].mean()), saved_global)
        if name == "universal v": check("§6 mean over 17 tiles, universal v", float(T.mean()), 0.6884909546375275)
        for k in range(50):
            rows.append({"condition": name, "image_id": IDS[k], "pos": k, **{f"tile_{t:02d}": float(T[k, t]) for t in range(T.shape[1])}})
    write_csv("C_tiles_split_on.csv", rows)


# ================= D. nb02: prompt table (100) and real-pipeline random noise (CUDA draw) =================
if RUN["D_prompts_nb02"]:
    print("\nD. nb02 re-run: four prompts after preprocessing (100 images), B1/B2 random noise (CUDA draw)")
    def yesno(s):
        w = s.strip().lower().lstrip("*\"' ").split()
        return w[0].strip(".,!") if w else ""
    xA = to01(E_u8[:100]); rows = []
    saved = {"describe (training)": (91, 2), "what is in it": (96, 12), "main object": (94, 28), "animal yes/no": (80, 3)}
    for pname, prompt in PROMPTS.items():
        c0 = gen_tensor(xA, prompt)
        res = {}
        for cond, pert in [("universal v", V10.to(DEVICE)), ("random (CUDA draw)", RAND_CUDA)]:
            c1 = gen_tensor((xA + pert).clamp(0, 1), prompt)
            if pname == "animal yes/no":
                res[cond] = 100 * np.mean([yesno(a) != yesno(b) for a, b in zip(c0, c1)])
            else:
                res[cond] = changed(c0, c1)
            rows += [{"prompt": pname, "condition": cond, "image_id": IDS[k], "pos": k, "clean": c0[k], "perturbed": c1[k]} for k in range(100)]
        print(f"  {pname}: v {res['universal v']:.0f}% (saved {saved[pname][0]}%), random {res['random (CUDA draw)']:.0f}% (saved {saved[pname][1]}%)")
    write_csv("D_prompts_after_preproc.csv", rows)
    rows = []
    for label, split, n, saved_r in [("B1 splitting off", False, 200, 3.5), ("B2 splitting on", True, 50, 2.0)]:
        c0 = gen_pipeline(E_u8[:n], P_DESCRIBE, None, split)
        c1 = gen_pipeline(E_u8[:n], P_DESCRIBE, RAND_CUDA, split)
        print(f"  {label}: random (CUDA draw) {changed(c0, c1):.1f}% (saved {saved_r}%)")
        rows += [{"setting": label, "condition": "random (CUDA draw)", "image_id": IDS[k], "pos": k, "clean": c0[k], "perturbed": c1[k]} for k in range(n)]
    write_csv("D_pipeline_random_cuda.csv", rows)


# ================= E. nb05 defenses without saved captions: JPEG, blur (200), blanked (100) =================
if RUN["E_defenses_nb05"]:
    print("\nE. nb05 re-run: JPEG q75, blur σ1 (200 images), global tile blanked (100 images)")
    GRAY = (128, 128, 128)
    def jpeg_pil(p, rng=None):
        buf = io.BytesIO(); p.save(buf, format="JPEG", quality=75); buf.seek(0); return Image.open(buf).convert("RGB")
    def blur_pil(p, rng=None): return p.filter(ImageFilter.GaussianBlur(radius=1.0))
    GRAY_TILE = processor.image_processor(images=[[Image.new("RGB", (IMG_SIZE, IMG_SIZE), GRAY)]],
                                          do_image_splitting=False, return_tensors="pt")["pixel_values"][0, 0]
    DEFS = {"none (splitting off)": {"split": False, "pil": None, "global": None},
            "none (splitting on)": {"split": True, "pil": None, "global": None},
            "JPEG q75": {"split": False, "pil": jpeg_pil, "global": None},
            "blur σ1": {"split": False, "pil": blur_pil, "global": None},
            "global tile: blanked": {"split": True, "pil": None, "global": "blank"}}
    @torch.no_grad()
    def gen_def(u8s, pert, d, bs=8):                                       # nb05 gen()
        outs = []
        for i in range(0, len(u8s), bs):
            pils = [to_pil_u8(u, pert) for u in u8s[i:i+bs]]
            if d["pil"] is not None: pils = [d["pil"](p) for p in pils]
            inp = processor(text=[chat(P_DESCRIBE)] * len(pils), images=[[p] for p in pils],
                            do_image_splitting=d["split"], return_tensors="pt", padding=True)
            if d["global"] == "blank":
                for j in range(len(pils)): inp["pixel_values"][j, -1] = GRAY_TILE
            inp = inp.to(DEVICE)
            g = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            outs += processor.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return [o.strip() for o in outs]
    saved = {"JPEG q75": (3.0, 8.5, 23.0), "blur σ1": (6.0, 5.0, 29.5), "global tile: blanked": (23.0, 20.0, 80.0)}
    adapt = {"JPEG q75": "adaptive_v10_JPEG_q75.npy", "blur σ1": "adaptive_v10_blur_σ1.npy",
             "global tile: blanked": "adaptive_v10_global_tile__blanked.npy"}
    REF = {}; rows = []
    for name in ["JPEG q75", "blur σ1", "global tile: blanked"]:
        d = DEFS[name]; n = 100 if d["split"] else 200; key = "on" if d["split"] else "off"
        if key not in REF: REF[key] = gen_def(E_u8[:n], None, DEFS[f"none (splitting {key})"])
        caps = {"undefended clean": REF[key], "clean, defended": gen_def(E_u8[:n], None, d),
                "fixed v, defended": gen_def(E_u8[:n], V10, d), "weak adaptive v, defended": gen_def(E_u8[:n], load_v(adapt[name]), d)}
        got = (changed(caps["undefended clean"], caps["clean, defended"]), changed(caps["clean, defended"], caps["fixed v, defended"]),
               changed(caps["clean, defended"], caps["weak adaptive v, defended"]))
        print(f"  {name}: clean cost {got[0]:.1f}% / fixed {got[1]:.1f}% / adaptive {got[2]:.1f}%   saved {saved[name]}")
        for cond, cl in caps.items():
            rows += [{"defense": name, "condition": cond, "image_id": IDS[k], "pos": k, "caption": cl[k]} for k in range(n)]
    write_csv("E_defenses_nb05.csv", rows)


# ================= F. nb03-F transfer: 500M (200 images) and 2.2B (100 images) =================
if RUN["F_transfer_nb03"]:
    print("\nF. transfer to SmolVLM-500M and SmolVLM-2.2B (real pipeline, splitting off)")
    del model; gc.collect(); torch.cuda.empty_cache()
    rows_c, rows_t = [], []
    saved = {"500M": (31.0, 10.0, 0.7060839533805847), "2.2B": (7.0, 3.0, 0.8713474869728088)}
    for key, mid, n, dtype in [("500M", "HuggingFaceTB/SmolVLM-500M-Instruct", 200, torch.float32),
                               ("2.2B", "HuggingFaceTB/SmolVLM-Instruct", 100, torch.float16)]:
        proc = AutoProcessor.from_pretrained(mid); proc.tokenizer.padding_side = "left"
        mdl = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=dtype).to(DEVICE).eval()
        c0 = gen_pipeline(E_u8[:n], P_DESCRIBE, None, False, proc=proc, mdl=mdl)
        rates = []
        for cond, pert in [("universal v", V10), ("random (CPU draw)", RAND_CPU[10])]:
            c1 = gen_pipeline(E_u8[:n], P_DESCRIBE, pert, False, proc=proc, mdl=mdl); rates.append(changed(c0, c1))
            rows_c += [{"model": key, "condition": cond, "image_id": IDS[k], "pos": k, "clean": c0[k], "perturbed": c1[k]} for k in range(n)]
            cos = tile_cosines(E_u8[:30], pert, False, proc=proc, mdl=mdl)[:, 0].numpy()
            if cond == "universal v": check(f"§8 token cos {key}", float(cos.mean()), saved[key][2])
            rows_t += [{"model": key, "condition": cond, "image_id": IDS[k], "pos": k, "token_cos": float(cos[k])} for k in range(30)]
        print(f"  {key}: v {rates[0]:.1f}% (saved {saved[key][0]}), random {rates[1]:.1f}% (saved {saved[key][1]})")
        del mdl, proc; gc.collect(); torch.cuda.empty_cache()
    write_csv("F_transfer_captions.csv", rows_c); write_csv("F_transfer_tokcos.csv", rows_t)

print("\nDone. Copy", OUT, "to stats_audit/reeval/ and rerun audit.py.")
