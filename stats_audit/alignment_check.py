"""Can held-out position i be used as the image ID across all saved caption files?

No caption file stores an image ID. Every notebook builds the held-out set the same way
(sorted file list, RandomState(0) permutation, held-out = perm[500:1000], n-image runs = first n),
so position i should be the same image everywhere. This script tests that claim four ways:

  A. exact identity  - runs in the same setting must give identical captions at every position
  B. saved examples  - the few example captions nb02 / nb03-F kept must sit at positions 0..3 of the other files
  C. shift test      - across settings, caption i of list X must be most similar to caption i of list Y,
                       not to caption i+k (a shift of k would move the peak to k)
  D. ground truth    - caption i must match the ImageNet class of the file at position i
                       (file list rebuilt by build_file_list.py), better than shifted positions do

Run with the audit venv:  C:/Users/kathi/.venvs/uap_audit/Scripts/python.exe alignment_check.py
CPU only; loads MiniLM (all-MiniLM-L6-v2, same checkpoint as the notebooks), never SmolVLM.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "smol_exp" / "uap_smolvlm"
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
SYNSETS = ROOT.parent / "Experiments ViT" / "python" / "data" / "synset_words.txt"
SHIFTS = list(range(-10, 11))

def L(name): return json.load(open(DATA / name, encoding="utf-8"))
w1 = L("week1_results.json")["captions"]
s1 = L("week2_step1_results.json")["examples"]
s2 = L("week2_step2_results.json")
C, E, F = s2["C_dominant"]["all_captions"], s2["E_short"], s2["F_transfer"]
sw = L("week2_step3_eps_sweep.json")["captions"]
sa = L("week3_strong_adaptive.json")["captions"]
ft = L("week3_finetune_and_rrp.json")["captions"]

st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
def emb(xs): return st.encode(list(xs), normalize_embeddings=True, convert_to_numpy=True, batch_size=64)

report = {}

# ---------- A. exact identity ----------
def ident(a, b):
    n = min(len(a), len(b)); return int(sum(x == y for x, y in zip(a[:n], b[:n]))), n
A = {
    "clean: nb03-C vs nb04": ident(C["clean"], sw["clean"]),
    "clean: nb04 vs nb07": ident(sw["clean"], ft["clean, original encoder"]),
    "v eps10: nb03-C vs nb04": ident(C["universal v"], sw["universal v eps10"]),
    "v eps10: nb04 vs nb07 (fixed v, no defense)": ident(sw["universal v eps10"], ft["fixed v, no defense"]),
    "random eps10 (CPU draw): nb03-C vs nb04": ident(C["random ±ξ"], sw["random ±ξ eps10"]),
}
report["A_exact_identity"] = {k: f"{m}/{n}" for k, (m, n) in A.items()}

# ---------- B. saved examples sit at positions 0..k ----------
B = {}
def ex_check(label, pairs, clean_list, pert_list=None):
    hits = []
    for j, (c0, c1) in enumerate(pairs):
        ok = clean_list[j] == c0 and (pert_list is None or pert_list[j] == c1)
        hits.append(ok)
    B[label] = f"{sum(hits)}/{len(hits)} example pairs found at the same positions"
ex_check("nb02 A describe (after preproc), universal v -> week1", s1["A|describe (week 1)|universal v"], w1["clean"], w1["universal v"])
ex_check("nb02 B1 (real, off), universal v -> nb04 eps10", s1["B|B1 splitting off|universal v"], sw["clean"], sw["universal v eps10"])
ex_check("nb02 B2 (real, on), universal v -> nb06 fixed v, no defense", s1["B|B2 splitting on (default)|universal v"], sa["clean"], sa["fixed v, no defense"])
ex_check("nb02 B2 (real, on), clean only -> nb06 clean", s1["B|B2 splitting on (default)|universal v"], sa["clean"])
report["B_saved_examples"] = B

# ---------- C. shift test across settings ----------
def shift_test(x, y):
    n = min(len(x), len(y)); S = emb(x[:n]) @ emb(y[:n]).T
    by_shift = {}
    for k in SHIFTS:
        idx = [(i, i + k) for i in range(n) if 0 <= i + k < n]
        by_shift[k] = float(np.mean([S[i, j] for i, j in idx]))
    top1 = float(np.mean(S.argmax(1) == np.arange(n)))
    best_k = max(by_shift, key=by_shift.get)
    others = [v for k, v in by_shift.items() if k != 0]
    return {"n": n, "mean_sim_shift0": round(by_shift[0], 3), "best_shift": best_k,
            "mean_sim_other_shifts": round(float(np.mean(others)), 3), "max_other_shift": round(max(others), 3),
            "top1_same_position": round(top1, 3)}
report["C_shift_test"] = {
    "after-preproc clean (nb01) vs real-pipeline clean (nb04)": shift_test(w1["clean"], sw["clean"]),
    "splitting-on clean (nb06) vs splitting-off clean (nb04)": shift_test(sa["clean"], sw["clean"]),
    "nb06 clean, defended vs nb06 clean": shift_test(sa["clean, defended"], sa["clean"]),
    "short answers (nb03-E) vs describe captions (nb04)": shift_test(E["clean_answers"], sw["clean"][:100]),
    "FT-fixed clean (nb07) vs original clean (nb07)": shift_test(ft["clean, FT-fixed"], ft["clean, original encoder"]),
    "FT-UAT clean (nb07) vs original clean (nb07)": shift_test(ft["clean, FT-UAT"], ft["clean, original encoder"]),
    "rrp clean (nb07) vs original clean (nb07)": shift_test(ft["clean, rrp defense"], ft["clean, original encoder"]),
}

# ---------- D. ground truth: ImageNet class of the file at each position ----------
fl = ROOT / "data" / "heldout_files.json"
if fl.exists():
    held = json.load(open(fl))["held_out"]
    names = [l.split(" ", 1)[1].strip() for l in open(SYNSETS, encoding="utf-8")]
    short = [n.split(",")[0] for n in names]
    cls = [h["class_idx"] for h in held[:200]]
    def class_test(caps, label):
        n = len(caps)
        Ec, Ek = emb(caps), emb([f"a photo of a {s}" for s in short])          # 1000 class prompts
        S = Ec @ Ek.T                                                            # (n, 1000)
        rank = lambda i, c: int((S[i] > S[i, c]).sum()) + 1                       # rank of class c among 1000
        by_shift = {}
        for k in SHIFTS:
            rs = [rank(i, cls[i + k]) for i in range(n) if 0 <= i + k < n]
            by_shift[k] = {"median_rank": float(np.median(rs)), "top10": float(np.mean(np.array(rs) <= 10))}
        others = [v["top10"] for k, v in by_shift.items() if k != 0]
        return {"label": label, "n": n, "shift0": by_shift[0],
                "best_shift_by_top10": max(by_shift, key=lambda k: by_shift[k]["top10"]),
                "other_shifts_top10_mean": round(float(np.mean(others)), 3),
                "other_shifts_top10_max": round(max(others), 3),
                "chance_top10": 0.01,
                "first5": [{"pos": i, "class": short[cls[i]], "caption": caps[i][:70]} for i in range(5)]}
    report["D_ground_truth_class"] = [
        class_test(sw["clean"], "nb04 clean (real pipeline, off)"),
        class_test(w1["clean"], "nb01 clean (after preprocessing)"),
        class_test(sa["clean"], "nb06 clean (splitting on)"),
        class_test(E["clean_answers"], "nb03-E short answers"),
    ]
    report["D_examples_2.2B"] = [{"pos": i, "class": short[cls[i]], "caption_2.2B": c0[:70]}
                                 for i, (c0, _) in enumerate(F["2.2B"]["universal v"]["examples"])]
else:
    report["D_ground_truth_class"] = "skipped: data/heldout_files.json not found (run build_file_list.py)"

json.dump(report, open(OUT / "alignment_check.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))
