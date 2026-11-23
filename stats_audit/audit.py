"""Statistical audit of the draft "Universal Adversarial Perturbations on a Tiny Vision-Language Model".

Runs from scratch on the saved files in smol_exp/uap_smolvlm/ (read only). CPU only: it loads MiniLM
(sentence-transformers/all-MiniLM-L6-v2, the checkpoint every notebook uses) to score saved captions.
It never loads SmolVLM and never trains anything.

    C:/Users/kathi/.venvs/uap_audit/Scripts/python.exe audit.py

Outputs (stats_audit/):
    all_numbers.csv            one row per number reported in the draft
    paired_tests.csv           exact McNemar tests for the comparisons in scope
    results/similarities.csv   per-image caption similarity for every scored condition
    results/near_threshold.csv pairs within 0.001 of 0.5 or 0.7 (CPU vs GPU rounding could flip them)
    results/step6.json         extra numbers from saved data (Step 6)

Image identity: no caption file stores an image ID. image_id is the held-out file name rebuilt by
build_file_list.py (data/heldout_files.json) when available, otherwise "heldout_pos_XXX". Both encode the
same thing: held-out position i = perm[500 + i] in every notebook (checked by alignment_check.py).
"""
import csv, json, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "smol_exp" / "uap_smolvlm"
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
THR, THR_STRICT = 0.5, 0.7


# ----------------------------------------------------------------------------------------------
# The two shared helpers, exactly as specified
# ----------------------------------------------------------------------------------------------
def rate(changed):
    """Percentage of 1s with its 95% Wilson interval."""
    changed = np.asarray(changed, int)
    k, n = int(changed.sum()), len(changed)
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return 100 * k / n, 100 * ci.low, 100 * ci.high, n

def mcnemar(a, b):
    """Two 0/1 arrays aligned on image_id: difference in points, discordant counts, exact McNemar p."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return 100 * (a - b).mean(), only_a, only_b, p


# ----------------------------------------------------------------------------------------------
# Saved captions, keyed by image_id
# ----------------------------------------------------------------------------------------------
def L(name): return json.load(open(DATA / name, encoding="utf-8"))
W1 = L("week1_results.json")
S1 = L("week2_step1_results.json")
S2 = L("week2_step2_results.json")
SW = L("week2_step3_eps_sweep.json")
DEF = L("week3_defenses_testtime.json")
SA = L("week3_strong_adaptive.json")
FT = L("week3_finetune_and_rrp.json")

_fl = ROOT / "data" / "heldout_files.json"
if _fl.exists():
    _held = json.load(open(_fl))["held_out"]
    IMAGE_ID = [h["file"] for h in _held]
    ID_SOURCE = "file name rebuilt from the ImageNetV2 archive (data/heldout_files.json)"
else:
    IMAGE_ID = [f"heldout_pos_{i:03d}" for i in range(500)]
    ID_SOURCE = "held-out position (perm[500 + i])"

def keyed(captions):
    """list (held-out order) -> {image_id: caption}"""
    return {IMAGE_ID[i]: c for i, c in enumerate(captions)}

CAP = {  # run name -> {image_id: caption}; file recorded for provenance
    "nb01/clean": (keyed(W1["captions"]["clean"]), "week1_results.json"),
    "nb01/v": (keyed(W1["captions"]["universal v"]), "week1_results.json"),
    "nb01/random": (keyed(W1["captions"]["random ±ξ"]), "week1_results.json"),
    "nb01/googlenet": (keyed(W1["captions"]["replication v (GoogLeNet, upsampled)"]), "week1_results.json"),
    "nb03C/clean": (keyed(S2["C_dominant"]["all_captions"]["clean"]), "week2_step2_results.json"),
    "nb03C/v": (keyed(S2["C_dominant"]["all_captions"]["universal v"]), "week2_step2_results.json"),
    "nb03C/random": (keyed(S2["C_dominant"]["all_captions"]["random ±ξ"]), "week2_step2_results.json"),
    "nb03E/clean": (keyed(S2["E_short"]["clean_answers"]), "week2_step2_results.json"),
    "nb03E/v": (keyed(S2["E_short"]["universal v"]["answers"]), "week2_step2_results.json"),
    "nb03E/random": (keyed(S2["E_short"]["random ±ξ"]["answers"]), "week2_step2_results.json"),
    "nb04/clean": (keyed(SW["captions"]["clean"]), "week2_step3_eps_sweep.json"),
}
for e in (4, 8, 10, 16):
    CAP[f"nb04/v{e}"] = (keyed(SW["captions"][f"universal v eps{e}"]), "week2_step3_eps_sweep.json")
    CAP[f"nb04/random{e}"] = (keyed(SW["captions"][f"random ±ξ eps{e}"]), "week2_step3_eps_sweep.json")
for k in SA["captions"]:
    CAP[f"nb06/{k}"] = (keyed(SA["captions"][k]), "week3_strong_adaptive.json")
for k in FT["captions"]:
    CAP[f"nb07/{k}"] = (keyed(FT["captions"][k]), "week3_finetune_and_rrp.json")

# Captions from the Colab re-run (reeval_colab.py) of the runs that never saved theirs (nb02, nb03-F, nb05).
# Every CSV row carries its image_id; the re-run is caption-for-caption identical to the saved runs wherever
# they overlap (checked in reeval_check() below).
REEVAL = ROOT / "reeval" / "stats_audit_reeval"
if not REEVAL.exists():
    REEVAL = ROOT / "reeval"
def _rcsv(name):
    p = REEVAL / name
    return (list(csv.DictReader(open(p, encoding="utf-8"))), f"reeval/{name}") if p.exists() else (None, None)
def _put(run, rows, col, src):
    CAP[run] = ({r["image_id"]: r[col] for r in rows}, src)

_d, _src = _rcsv("D_prompts_after_preproc.csv")
if _d:
    for pn in {r["prompt"] for r in _d}:
        rv = [r for r in _d if r["prompt"] == pn and r["condition"] == "universal v"]
        rr = [r for r in _d if r["prompt"] == pn and r["condition"] == "random (CUDA draw)"]
        _put(f"nb02A/{pn}/clean", rv, "clean", _src); _put(f"nb02A/{pn}/v", rv, "perturbed", _src)
        _put(f"nb02A/{pn}/random", rr, "perturbed", _src)
_d, _src = _rcsv("D_pipeline_random_cuda.csv")
if _d:
    for s in ("B1 splitting off", "B2 splitting on"):
        rs = [r for r in _d if r["setting"] == s]
        _put(f"nb02/{s}/clean", rs, "clean", _src); _put(f"nb02/{s}/random", rs, "perturbed", _src)
_d, _src = _rcsv("E_defenses_nb05.csv")
if _d:
    for dn, cn in {(r["defense"], r["condition"]) for r in _d}:
        _put(f"nb05/{dn}/{cn}", [r for r in _d if r["defense"] == dn and r["condition"] == cn], "caption", _src)
_d, _src = _rcsv("F_transfer_captions.csv")
if _d:
    for m in ("500M", "2.2B"):
        rv = [r for r in _d if r["model"] == m and r["condition"] == "universal v"]
        rr = [r for r in _d if r["model"] == m and r["condition"] == "random (CPU draw)"]
        _put(f"nb03F/{m}/clean", rv, "clean", _src); _put(f"nb03F/{m}/v", rv, "perturbed", _src)
        _put(f"nb03F/{m}/random", rr, "perturbed", _src)

def reeval_check():
    """Re-run captions vs saved captions for the same setting, merged on image_id."""
    out = {}
    for label, a, b in [("nb02 describe, clean vs nb01 clean", "nb02A/describe (training)/clean", "nb01/clean"),
                        ("nb02 describe, universal v vs nb01", "nb02A/describe (training)/v", "nb01/v"),
                        ("nb02 describe, random (CUDA) vs nb01", "nb02A/describe (training)/random", "nb01/random"),
                        ("nb02 B1 clean vs nb04 clean", "nb02/B1 splitting off/clean", "nb04/clean"),
                        ("nb02 B2 clean vs nb06 clean", "nb02/B2 splitting on/clean", "nb06/clean"),
                        ("nb05 JPEG undefended clean vs nb04 clean", "nb05/JPEG q75/undefended clean", "nb04/clean"),
                        ("nb05 blanked undefended clean vs nb06 clean", "nb05/global tile: blanked/undefended clean", "nb06/clean")]:
        if a in CAP and b in CAP:
            x, y = CAP[a][0], CAP[b][0]; sh = [i for i in IMAGE_ID if i in x and i in y]
            out[label] = f"{sum(x[i] == y[i] for i in sh)}/{len(sh)} identical"
    return out


# ----------------------------------------------------------------------------------------------
# Caption similarity (MiniLM, CPU), cached per unique string
# ----------------------------------------------------------------------------------------------
_st, _emb = None, {}
def _encode(texts):
    global _st
    new = [t for t in set(texts) if t not in _emb]
    if new:
        if _st is None:
            from sentence_transformers import SentenceTransformer
            _st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        for t, v in zip(new, _st.encode(new, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)):
            _emb[t] = v

SIM_ROWS, NEAR = [], []
def scored(ref, cond, first=None):
    """Merge two runs on image_id (optionally only the first n held-out images) and score each pair.
    Returns (image_ids, similarities)."""
    a, fa = CAP[ref]; b, fb = CAP[cond]
    ids = [i for i in IMAGE_ID if i in a and i in b]
    if first is not None:
        ids = [i for i in ids if IMAGE_ID.index(i) < first]
    _encode([a[i] for i in ids] + [b[i] for i in ids])
    s = np.array([float(_emb[a[i]] @ _emb[b[i]]) for i in ids])
    tag = f"{ref} -> {cond}"
    if not any(r["pair"] == tag for r in SIM_ROWS):
        for i, x in zip(ids, s):
            SIM_ROWS.append({"pair": tag, "image_id": i, "pos": IMAGE_ID.index(i), "similarity": round(x, 6),
                             "ref_caption": a[i], "caption": b[i]})
            for t in (THR, THR_STRICT):
                if abs(x - t) < 1e-3:
                    NEAR.append({"pair": tag, "image_id": i, "threshold": t, "similarity": round(x, 6)})
    return ids, s

def flags(ref, cond, thr=THR, first=None):
    ids, s = scored(ref, cond, first)
    return ids, (s < thr).astype(int)

def source(*runs): return "; ".join(sorted({CAP[r][1] for r in runs}))


# ----------------------------------------------------------------------------------------------
# all_numbers.csv
# ----------------------------------------------------------------------------------------------
ROWS = []
def _round_like(x, draft):
    """Round x the way the draft prints `draft` (same number of decimals, half up)."""
    d = draft.strip().rstrip("%")
    places = len(d.split(".")[1]) if "." in d else 0
    q = Decimal(1).scaleb(-places)
    return Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP), Decimal(d)

def add_rate(section, desc, draft, ref, cond, thr=THR, first=None, evidence="verified", note=""):
    ids, f = flags(ref, cond, thr, first)
    r, lo, hi, n = rate(f)
    got, want = _round_like(r, draft)
    ROWS.append({"section": section, "description": desc, "draft_value": draft,
                 "recomputed_value": f"{r:.1f}%", "n_images": n, "ci_low": f"{lo:.1f}", "ci_high": f"{hi:.1f}",
                 "sd": "", "matches_draft": "yes" if got == want else "no", "evidence": evidence,
                 "source_file": source(ref, cond), "note": note})
    return ids, f

def add_value(section, desc, draft, recomputed, n, src, evidence, matches, sd="", lo="", hi="", note=""):
    ROWS.append({"section": section, "description": desc, "draft_value": draft, "recomputed_value": recomputed,
                 "n_images": n, "ci_low": lo, "ci_high": hi, "sd": sd, "matches_draft": matches,
                 "evidence": evidence, "source_file": src, "note": note})

def add_untraceable(section, desc, draft, n, src, note, saved=None):
    """Aggregate-only number: Wilson interval from the saved rate (k = rate * n), flagged NOT TRACEABLE."""
    lo = hi = ""
    if saved is not None and draft.endswith("%"):
        k = int(round(saved * n)); _, l, h, _ = rate([1] * k + [0] * (n - k)); lo, hi = f"{l:.1f}", f"{h:.1f}"
    add_value(section, desc, draft, f"{100 * saved:.1f}% (saved aggregate)" if saved is not None else "",
              n, src, "aggregate only", "not traceable", lo=lo, hi=hi, note=note)

def rerun_or_untraceable(section, desc, draft, ref, cond, n, src, note, saved):
    """A rate whose captions were not saved: from the Colab re-run captions if present, else NOT TRACEABLE."""
    if ref in CAP and cond in CAP:
        return add_rate(section, desc, draft, ref, cond, evidence="re-evaluated (Colab re-run)",
                        note=f"{note}; re-run captions; saved aggregate {100 * saved:.1f}%")
    add_untraceable(section, desc, draft, n, src, note, saved)

def reeval_values(file, value, **filters):
    """Per-image values from a reeval CSV, or None if that file is not there yet.
    `value` is a column name or a function of the row (e.g. a mean over tile columns)."""
    path = REEVAL / file
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if all(r[k] == str(v) for k, v in filters.items())]
    return np.array([value(r) if callable(value) else float(r[value]) for r in rows]) if rows else None

def add_mean(section, desc, draft, saved, n, src, note="", per_image=None):
    """A mean (token cosine, PSNR). With per-image re-evaluation data: mean, SD and n from it.
    Without: the saved aggregate, SD 'missing'."""
    vals = reeval_values(*per_image[:2], **per_image[2]) if per_image else None
    if vals is not None:
        m = float(vals.mean()); got, want = _round_like(m, draft)
        add_value(section, desc, draft, f"{m:.3f}", len(vals), f"reeval/{per_image[0]} (saved aggregate {saved:.3f})",
                  "verified (re-evaluated)", "yes" if got == want else "no", sd=f"{vals.std(ddof=1):.3f}", note=note)
        return
    got, want = _round_like(saved, draft)
    add_value(section, desc, draft, f"{saved:.3f}", n, src, "aggregate only (no per-image values)",
              "yes" if got == want else "no", sd="missing", note=note or "SD needs per-image re-evaluation (reeval_colab.py)")

LOCAL = lambda r: np.mean([float(r[f"tile_{t:02d}"]) for t in range(16)])
GLOBAL = lambda r: float(r["tile_16"])
ALLTILES = lambda r: np.mean([float(r[f"tile_{t:02d}"]) for t in range(17)])


# ---------------- §5 / abstract: after preprocessing ----------------
add_rate("§5 table 1", "universal v, after preprocessing, captions changed", "91.0%", "nb01/clean", "nb01/v")
add_rate("§5 table 1", "random ±ξ (CUDA draw), after preprocessing", "5.0%", "nb01/clean", "nb01/random")
add_rate("§5 table 1", "GoogLeNet UAP upsampled, after preprocessing", "4.0%", "nb01/clean", "nb01/googlenet")
ev = W1["results"]["eval"]
add_mean("§5 table 1", "token cosine, universal v (after preprocessing)", "-0.051", ev["universal v"]["mean_token_cos"], 500, "week1_results.json",
         per_image=("A_tokcos_after_preproc.csv", "token_cos", {"condition": "universal v"}))
add_mean("§5 table 1", "token cosine, random ±ξ", "0.846", ev["random ±ξ"]["mean_token_cos"], 500, "week1_results.json",
         per_image=("A_tokcos_after_preproc.csv", "token_cos", {"condition": "random (CUDA draw)"}))
add_mean("§5 table 1", "token cosine, GoogLeNet UAP", "0.850", ev["replication v (GoogLeNet, upsampled)"]["mean_token_cos"], 500, "week1_results.json",
         per_image=("A_tokcos_after_preproc.csv", "token_cos", {"condition": "GoogLeNet v"}))
_, f = flags("nb01/clean", "nb01/v"); r, lo, hi, n = rate(1 - f)
add_value("§5 text", "captions that survived v (after preprocessing)", "9%", f"{r:.1f}%", n, "week1_results.json",
          "verified", "yes" if _round_like(r, "9%")[0] == 9 else "no", lo=f"{lo:.1f}", hi=f"{hi:.1f}")

# ---------------- §5 prompt table (nb02, captions not saved) ----------------
A = S1["results"]["A_prompts"]
add_rate("§5 prompt table", "describe, universal v (stand-in: nb01 captions, first 100)", "91%", "nb01/clean", "nb01/v", first=100,
         evidence="stand-in", note="nb02 captions not saved; nb01 used the same path, same v and same CUDA noise on the same first 100 images"
         + ("; the Colab re-run of nb02 gives the identical 100 captions" if "nb02A/describe (training)/v" in CAP else ""))
add_rate("§5 prompt table", "describe, random ±ξ (stand-in: nb01 captions, first 100)", "2%", "nb01/clean", "nb01/random", first=100,
         evidence="stand-in", note="same as above")
for pname, label, dv, dr in [("what is in it", "\"What is in this image?\"", "96%", "12%"),
                             ("main object", "\"What is the main object?\"", "94%", "28%")]:
    rerun_or_untraceable("§5 prompt table", f"{label}, universal v", dv, f"nb02A/{pname}/clean", f"nb02A/{pname}/v", 100,
                         "week2_step1_results.json", "nb02 captions not saved", A[pname]["universal v"]["changed"])
    rerun_or_untraceable("§5 prompt table", f"{label}, random ±ξ", dr, f"nb02A/{pname}/clean", f"nb02A/{pname}/random", 100,
                         "week2_step1_results.json", "nb02 captions not saved", A[pname]["random ±ξ"]["changed"])
yn = A["animal yes/no"]
def yesno(s):  # nb02 definition
    w = s.strip().lower().lstrip("*\"' ").split()
    return w[0].strip(".,!") if w else ""
YN = "nb02A/animal yes/no/"
for cond, lab in [("v", "universal v"), ("random", "random ±ξ")]:
    for what, sec, dv, fn in [("answers flipped", "§5 prompt table", {"v": "80%", "random": "3%"}[cond],
                               lambda a, b: int(yesno(a) != yesno(b))),
                              ("still a valid yes/no", "§5 text", {"v": "28%", "random": "100%"}[cond],
                               lambda a, b: int(yesno(b) in ("yes", "no")))]:
        saved = yn[lab]["answer_flipped" if what == "answers flipped" else "valid_yes_no_after"]
        if YN + cond in CAP:
            c0, c1 = CAP[YN + "clean"][0], CAP[YN + cond][0]; ids = [i for i in IMAGE_ID if i in c0 and i in c1]
            r, lo, hi, n = rate([fn(c0[i], c1[i]) for i in ids])
            add_value(sec, f"yes/no, {what}, {lab}", dv, f"{r:.1f}%", n, CAP[YN + cond][1], "re-evaluated (Colab re-run)",
                      "yes" if _round_like(r, dv)[0] == _round_like(r, dv)[1] else "no", lo=f"{lo:.1f}", hi=f"{hi:.1f}",
                      note=f"nb02 first-word rule; saved aggregate {100 * saved:.1f}%")
        else:
            add_untraceable(sec, f"yes/no, {what}, {lab}", dv, 100, "week2_step1_results.json", "nb02 answers not saved", saved)

# ---------------- §6 / §8 / §9 / abstract: real pipeline, splitting off ----------------
ids_off, f_off = add_rate("Abstract, §2.2, §6, §8, §9", "universal v, real pipeline, splitting off", "71.5%", "nb04/clean", "nb04/v10",
                          note="identical captions in nb03 part C and nb07 'fixed v, no defense'")
rerun_or_untraceable("§6, §8", "random ±ξ (CUDA draw, nb02), real pipeline, splitting off", "3.5%",
                     "nb02/B1 splitting off/clean", "nb02/B1 splitting off/random", 200, "week2_step1_results.json",
                     "nb02 used the CUDA noise draw and did not save captions; the saved CPU draw gives the §9 4.0%",
                     S1["results"]["B_pipeline"]["B1 splitting off"]["random ±ξ"]["changed"])
add_rate("Abstract ('at most 4%'), §9", "random ±ξ (CPU draw), real pipeline, splitting off, 10/255", "4.0%", "nb04/clean", "nb04/random10")
B = S1["results"]["B_pipeline"]
add_mean("§6, §8, §9", "token cosine, universal v, real pipeline off", "0.142", B["B1 splitting off"]["universal v"]["token_cos"], 50,
         "week2_step1_results.json", note="computed on the first 50 images although the §6 and §8 tables say 200",
         per_image=("B_tokcos_psnr_pipeline.csv", "token_cos", {"eps": 10, "condition": "universal v"}))
add_mean("§6", "token cosine, splitting on, averaged over 17 tiles", "0.688", B["B2 splitting on (default)"]["universal v"]["token_cos"], 50, "week2_step1_results.json",
         per_image=("C_tiles_split_on.csv", ALLTILES, {"condition": "universal v"}))
ids_on, f_on = add_rate("Abstract, §2.2, §4.5, §6, §11.2", "universal v, real pipeline, splitting on (121/200)", "60.5%", "nb06/clean", "nb06/fixed v, no defense")
add_rate("§6", "splitting on, first 50 images (stand-in: nb06 captions)", "70.0%", "nb06/clean", "nb06/fixed v, no defense", first=50,
         evidence="stand-in", note="nb02 B2 captions not saved")
rerun_or_untraceable("§6", "random ±ξ, splitting on, 50 images", "2.0%", "nb02/B2 splitting on/clean", "nb02/B2 splitting on/random", 50,
                     "week2_step1_results.json", "no random-noise captions with splitting on were saved",
                     B["B2 splitting on (default)"]["random ±ξ"]["changed"])
add_rate("§6, §11.1", "splitting on, first 100 images (stand-in: nb06 captions)", "59.0%", "nb06/clean", "nb06/fixed v, no defense", first=100,
         evidence="stand-in", note="nb05 captions not saved")

# attractor (§6), nb03 definitions
C = S2["C_dominant"]
OPENER = re.compile(r"^(in this (image|picture)|this image (shows|is))[, ]*((i|we) can see|there (is|are)|we see)?[, ]*")
def norm_cap(s): return OPENER.sub("", re.sub(r"[^a-z0-9 ,]", "", s.lower()).strip()).strip(" ,")
def person3(c): c = c.lower(); return int("person" in c or "man " in c or "woman" in c)
for cond, dv in [("clean", "14%"), ("universal v", "28%")]:
    caps = C["all_captions"][cond]; r, lo, hi, n = rate([person3(c) for c in caps])
    add_value("§6", f"captions mentioning a person (person/man/woman), {cond}", dv, f"{r:.1f}%", n, "week2_step2_results.json",
              "verified", "yes" if _round_like(r, dv)[0] == Decimal(dv[:-1]) else "no", lo=f"{lo:.1f}", hi=f"{hi:.1f}")
n_dist = len({norm_cap(c) for c in C["all_captions"]["universal v"]})
add_value("§6", "distinct perturbed captions (nb03 normalisation)", "173/200", f"{n_dist}/200", 200, "week2_step2_results.json",
          "verified", "yes" if n_dist == 173 else "no")
ws = [int("white shirt" in c.lower()) for c in C["all_captions"]["universal v"]]; r, lo, hi, n = rate(ws)
add_value("§6", "perturbed captions containing 'white shirt'", "roughly one in six (16.7%)", f"{r:.1f}%", n, "week2_step2_results.json",
          "verified", "yes" if abs(r - 100 / 6) < 2.5 else "no", lo=f"{lo:.1f}", hi=f"{hi:.1f}")

# ---------------- §7 tiles (aggregate only) ----------------
D = S2["D_tiles"]["mean_cos_per_tile"]
for cond, dl, dg in [("universal v", "0.723", "0.142"), ("random ±ξ", "0.771", "0.865")]:
    add_mean("§7", f"local tiles mean token cosine, {cond}", dl, float(np.mean(D[cond][:-1])), 50, "week2_step2_results.json",
             note="mean of 16 tile means",
             per_image=("C_tiles_split_on.csv", LOCAL, {"condition": "universal v" if cond == "universal v" else "random (CPU draw)"}))
    add_mean("§7", f"global tile token cosine, {cond}", dg, D[cond][-1], 50, "week2_step2_results.json",
             per_image=("C_tiles_split_on.csv", GLOBAL, {"condition": "universal v" if cond == "universal v" else "random (CPU draw)"}))

# ---------------- §8 transfer (captions not saved) ----------------
T = S2["F_transfer"]
for m, n, dv, dr in [("500M", 200, "31.0%", "10.0%"), ("2.2B", 100, "7.0%", "3.0%")]:
    rerun_or_untraceable("§8", f"{m}, universal v", dv, f"nb03F/{m}/clean", f"nb03F/{m}/v", n, "week2_step2_results.json",
                         "transfer captions not saved", T[m]["universal v"]["changed"])
    rerun_or_untraceable("§8", f"{m}, random ±ξ", dr, f"nb03F/{m}/clean", f"nb03F/{m}/random", n, "week2_step2_results.json",
                         "transfer captions not saved", T[m]["random ±ξ"]["changed"])
add_mean("§8", "500M token cosine, universal v", "0.706", T["500M"]["universal v"]["token_cos"], 30, "week2_step2_results.json", note="computed on 30 images, not stated in the draft",
         per_image=("F_transfer_tokcos.csv", "token_cos", {"model": "500M", "condition": "universal v"}))
add_mean("§8", "2.2B token cosine, universal v", "0.871", T["2.2B"]["universal v"]["token_cos"], 30, "week2_step2_results.json", note="computed on 30 images, not stated in the draft",
         per_image=("F_transfer_tokcos.csv", "token_cos", {"model": "2.2B", "condition": "universal v"}))

# ---------------- §9 budget sweep ----------------
ROW = SW["rows"]
for e, dv, dr, dc, dp, dpsnr in [(4, "21.5%", "1.0%", "0.623", "14%", "37.0"), (8, "62.5%", "3.5%", "0.291", "22%", "32.0"),
                                 (10, "71.5%", "4.0%", "0.142", "28%", "30.1"), (16, "91.5%", "4.0%", "-0.054", "18%", "26.4")]:
    if e != 10:  # 10/255 rows already added above (same captions)
        add_rate("§9", f"universal v, {e}/255", dv, "nb04/clean", f"nb04/v{e}")
        add_rate("§9", f"random ±ξ, {e}/255", dr, "nb04/clean", f"nb04/random{e}")
    add_mean("§9", f"token cosine (real pipeline), universal v {e}/255", dc, ROW[str(e)]["universal v"]["token_cos_pipeline"], 50, "week2_step3_eps_sweep.json",
             per_image=("B_tokcos_psnr_pipeline.csv", "token_cos", {"eps": e, "condition": "universal v"}))
    caps = SW["captions"][f"universal v eps{e}"]
    r, lo, hi, n = rate([int("person" in c.lower() or "white shirt" in c.lower()) for c in caps])
    add_value("§9", f"captions with 'person' or 'white shirt', {e}/255", dp, f"{r:.1f}%", n, "week2_step3_eps_sweep.json",
              "verified", "yes" if _round_like(r, dp)[0] == Decimal(dp[:-1]) else "no", lo=f"{lo:.1f}", hi=f"{hi:.1f}",
              note="nb04 definition (person or white shirt), not the §6 one")
    add_mean("§9", f"PSNR (dB), universal v {e}/255", dpsnr, ROW[str(e)]["universal v"]["psnr_db"], 50, "week2_step3_eps_sweep.json",
             per_image=("B_tokcos_psnr_pipeline.csv", "psnr_db", {"eps": e, "condition": "universal v"}))

# ---------------- §10 / §12 short answers ----------------
add_rate("§10, §12", "short answers, random ±ξ, text similarity", "28%", "nb03E/clean", "nb03E/random")
add_rate("§10 (nb03 part E)", "short answers, universal v, text similarity", "83.0%", "nb03E/clean", "nb03E/v",
         note="printed by nb03; not quoted in the draft")
STOP = set("a an the of on in and with is are at to its it this that there some two one we can see i image picture which who".split())
def words(s): return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2]
def content(s): return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words(s)}
E = S2["E_short"]
cw = [int(not (content(x) & content(y))) for x, y in zip(E["clean_answers"], E["random ±ξ"]["answers"]) if content(x)]
r, lo, hi, n = rate(cw)
add_value("§10, §12", "short answers, random ±ξ, no shared content word", "28%", f"{r:.1f}%", n, "week2_step2_results.json",
          "verified", "yes" if _round_like(r, "28%")[0] == 28 else "no", lo=f"{lo:.1f}", hi=f"{hi:.1f}",
          note="answers with no content word in the clean answer are excluded, as in nb03")

# ---------------- §10 counts that do not need labels ----------------
key = list(csv.DictReader(open(DATA / "labeling_key_DO_NOT_OPEN_before_labeling.csv", encoding="utf-8")))
n_exact = sum(SW["captions"]["clean"][i] == SW["captions"][k][i] for k in SW["captions"] if k != "clean" for i in range(200))
add_value("§10", "unchanged pairs removed (exact string match) of 1,600", "377", str(n_exact), 1600, "week2_step3_eps_sweep.json", "verified", "yes" if n_exact == 377 else "no")
n_u = sum(r_["condition"].startswith("universal") for r_ in key)
add_value("§10", "labelled pairs: universal / random", "37 / 23", f"{n_u} / {len(key) - n_u}", 60, "labeling_key_DO_NOT_OPEN_before_labeling.csv",
          "verified", "yes" if (n_u, len(key) - n_u) == (37, 23) else "no")
for d, desc in [("86.4%", "agreement at 0.5"), ("73%", "universal pairs labelled changed (27/37)"), ("13%", "random pairs labelled changed (3/23)")]:
    add_value("§10", desc, d, "", "", "human labels (missing)", "no labels", "not traceable", note="Section 10 put aside: human labels not in the repository")

# ---------------- §11.1 (nb05, captions not saved; stand-ins where available) ----------------
for name, dc, df_, da in [("JPEG q75", "3.0%", "8.5%", "23.0%"), ("blur σ1", "6.0%", "5.0%", "29.5%"), ("global tile: blanked", "23.0%", "20.0%", "80.0%")]:
    r_ = DEF[name]; n = r_["n"]; P = f"nb05/{name}/"
    rerun_or_untraceable("§11.1", f"{name}: clean cost", dc, P + "undefended clean", P + "clean, defended", n,
                         "week3_defenses_testtime.json", "nb05 captions not saved", r_["clean_cost"])
    rerun_or_untraceable("§11.1", f"{name}: fixed v", df_, P + "clean, defended", P + "fixed v, defended", n,
                         "week3_defenses_testtime.json", "nb05 captions not saved", r_["fixed v (10/255)"])
    rerun_or_untraceable("§11.1", f"{name}: weak adaptive v", da, P + "clean, defended", P + "weak adaptive v, defended", n,
                         "week3_defenses_testtime.json", "nb05 captions not saved", r_["adaptive v (10/255)"])
add_rate("§11.1", "random resize-pad: clean cost (stand-in: nb07)", "6.0%", "nb07/clean, original encoder", "nb07/clean, rrp defense", evidence="stand-in", note="nb05 captions not saved")
add_rate("§11.1", "random resize-pad: fixed v (stand-in: nb07)", "4.0%", "nb07/clean, rrp defense", "nb07/fixed v, rrp defense", evidence="stand-in", note="nb05 captions not saved")
add_rate("§11.1", "random resize-pad: weak adaptive v (stand-in: nb07)", "21.0%", "nb07/clean, rrp defense", "nb07/weak adaptive v, rrp defense", evidence="stand-in", note="nb05 captions not saved")
add_rate("§11.1", "global tile resize-pad: clean cost, 100 images (stand-in: nb06 first 100)", "1.0%", "nb06/clean", "nb06/clean, defended", first=100, evidence="stand-in", note="nb05 captions not saved")
add_rate("§11.1", "global tile resize-pad: fixed v, 100 images (stand-in: nb06 first 100)", "2.0%", "nb06/clean, defended", "nb06/fixed v, defended", first=100, evidence="stand-in", note="nb05 captions not saved")
add_rate("§11.1", "global tile resize-pad: weak adaptive v, 100 images (stand-in: nb06 first 100)", "10.0%", "nb06/clean, defended", "nb06/weak adaptive v, defended", first=100, evidence="stand-in", note="nb05 captions not saved")

# ---------------- §11.2 ----------------
add_rate("§11.2", "global tile: clean cost (5/200)", "2.5%", "nb06/clean", "nb06/clean, defended")
add_rate("§11.2", "global tile: fixed v, defended", "1.5%", "nb06/clean, defended", "nb06/fixed v, defended")
ids_w, f_w = add_rate("§11.2", "global tile: weak adaptive v, defended", "13.0%", "nb06/clean, defended", "nb06/weak adaptive v, defended")
add_untraceable("§11.2", "global tile: strong adaptive v, defended, after 6 epochs", "30.5%", 200,
                "nb06 printed output only", "6-epoch captions and perturbation were overwritten by the 12-epoch rerun", 0.305)
ids_s, f_s = add_rate("§11.2, §11.4", "global tile: strong adaptive v (12 epochs), defended", "35.5%", "nb06/clean, defended", "nb06/strong adaptive v, defended")
ids_sn, f_sn = add_rate("§11.2", "strong adaptive v (12 epochs), no defense", "29.5%", "nb06/clean", "nb06/strong adaptive v, no defense")

# ---------------- §11.3 / §11.4 ----------------
for thr, dc in [(THR, "6.0%"), (THR_STRICT, "25.0%")]:
    add_rate("§11.4", f"random resize-pad: clean cost at {thr}", dc, "nb07/clean, original encoder", "nb07/clean, rrp defense", thr=thr)
add_rate("§11.4", "random resize-pad: strong adaptive v (8 epochs)", "29.0%", "nb07/clean, rrp defense", "nb07/strong adaptive v, rrp defense",
         note="Fig. 12 shows 28% (epoch 7); table and saved captions are epoch 8")
for var, dc5, dc7, dfix, dad in [("FT-fixed", "3.0%", "8.0%", "4.0%", "41.0%"), ("FT-UAT", "4.5%", "13.5%", "2.5%", "37.5%")]:
    add_rate("§11.3, §11.4", f"{var}: clean cost at 0.5", dc5, "nb07/clean, original encoder", f"nb07/clean, {var}")
    add_rate("§11.4", f"{var}: clean cost at 0.7", dc7, "nb07/clean, original encoder", f"nb07/clean, {var}", thr=THR_STRICT)
    add_rate("§11.3", f"{var}: fixed v", dfix, f"nb07/clean, {var}", f"nb07/fixed v, {var}")
    add_rate("§11.3, §11.4", f"{var}: adaptive v", dad, f"nb07/clean, {var}", f"nb07/adaptive v, {var}")
add_value("§11.3", "FT-fixed / FT-UAT clean-token drift", "0.971 / 0.953", "0.9706 / 0.9530 (printed)", 50, "nb07 printed output",
          "aggregate only", "yes", sd="missing")
add_value("§11.3", "attacker token cosine vs FT-UAT / FT-fixed", "0.675 / 0.361", "0.675 / 0.361 (printed training log)", 500,
          "nb07 printed output", "aggregate only", "yes", sd="missing", note="last-epoch training loss on the 500 BUILD images, not held-out images")
_tc = reeval_values("A_tokcos_after_preproc.csv", "token_cos", condition="random (CUDA draw)")
if _tc is not None:
    r, lo, hi, n = rate(_tc < 0.9)
    add_value("§12", "random noise passes token cosine < 0.9", "91.6%", f"{r:.1f}%", n, "reeval/A_tokcos_after_preproc.csv",
              "re-evaluated (Colab re-run)", "yes" if _round_like(r, "91.6%")[0] == Decimal("91.6") else "no", lo=f"{lo:.1f}", hi=f"{hi:.1f}",
              note=f"after preprocessing, 500 held-out images; saved aggregate {100 * ev['random ±ξ']['emb_fool@0.9']:.1f}%")
else:
    add_value("§12", "random noise passes token cosine < 0.9", "91.6%", f"{100 * ev['random ±ξ']['emb_fool@0.9']:.1f}% (saved aggregate)", 500,
              "week1_results.json", "aggregate only", "not traceable", note="after preprocessing, 500 held-out images; per-image cosines not saved")


# ----------------------------------------------------------------------------------------------
# paired_tests.csv (exact McNemar, merged on image_id)
# ----------------------------------------------------------------------------------------------
PT = []
def paired(comparison, label_a, ref_a, cond_a, label_b, ref_b, cond_b, thr=THR, confounded="no", draft="", note=""):
    ia, fa = flags(ref_a, cond_a, thr); ib, fb = flags(ref_b, cond_b, thr)
    A_, B_ = dict(zip(ia, fa)), dict(zip(ib, fb))
    shared = [i for i in IMAGE_ID if i in A_ and i in B_]               # merge on image_id
    d, oa, ob, p = mcnemar([A_[i] for i in shared], [B_[i] for i in shared])
    PT.append({"comparison": comparison, "condition_A": label_a, "condition_B": label_b, "n_paired": len(shared),
               "rate_A": f"{100 * np.mean([A_[i] for i in shared]):.1f}%", "rate_B": f"{100 * np.mean([B_[i] for i in shared]):.1f}%",
               "only_A": oa, "only_B": ob, "difference_points": f"{d:+.1f}", "p_value": f"{p:.4g}",
               "confounded": confounded, "draft_claim": draft, "note": note})

RRP = ("random resize-pad", "nb07/clean, rrp defense", "nb07/strong adaptive v, rrp defense")
FTU = ("FT-UAT", "nb07/clean, FT-UAT", "nb07/adaptive v, FT-UAT")
FTF = ("FT-fixed", "nb07/clean, FT-fixed", "nb07/adaptive v, FT-fixed")
GT = ("global tile resize-pad (splitting on)", "nb06/clean, defended", "nb06/strong adaptive v, defended")
claim_rank = "no single best defense (overlapping intervals)"
for (la, ra, ca), (lb, rb, cb) in [(RRP, FTU), (RRP, FTF), (FTU, FTF)]:
    paired("§11.4 strongest attack", la, ra, ca, lb, rb, cb, draft=claim_rank)
for la, ra, ca in [RRP, FTU, FTF]:
    paired("§11.4 strongest attack", GT[0], GT[1], GT[2], la, ra, ca, confounded="yes", draft=claim_rank,
           note="global tile runs with splitting on (baseline 60.5%), the others with splitting off (71.5%)")
CLEAN_OFF = {"random resize-pad": "nb07/clean, rrp defense", "FT-UAT": "nb07/clean, FT-UAT", "FT-fixed": "nb07/clean, FT-fixed"}
for thr in (THR, THR_STRICT):
    names = list(CLEAN_OFF)
    for a_i in range(3):
        for b_i in range(a_i + 1, 3):
            a, b = names[a_i], names[b_i]
            paired(f"§11.4 clean cost at {thr}", a, "nb07/clean, original encoder", CLEAN_OFF[a],
                   b, "nb07/clean, original encoder", CLEAN_OFF[b], thr=thr, draft="global tile is the cheapest on clean captions")
    for a in names:
        paired(f"§11.4 clean cost at {thr}", "global tile resize-pad (splitting on)", "nb06/clean", "nb06/clean, defended",
               a, "nb07/clean, original encoder", CLEAN_OFF[a], thr=thr, confounded="yes",
               draft="global tile is the cheapest on clean captions",
               note="clean cost measured against splitting-on clean captions for the global tile, splitting-off for the others")
paired("§11.2", "strong adaptive v (12 epochs), defended", "nb06/clean, defended", "nb06/strong adaptive v, defended",
       "weak adaptive v, defended", "nb06/clean, defended", "nb06/weak adaptive v, defended",
       draft="strong breaks 50 captions the weak did not; reverse 5")
PT.append({"comparison": "§11.2", "condition_A": "strong adaptive v, 6 epochs, defended", "condition_B": "strong adaptive v, 12 epochs, defended",
           "n_paired": "", "rate_A": "30.5% (printed)", "rate_B": "35.5%", "only_A": "", "only_B": "", "difference_points": "",
           "p_value": "not computable", "confounded": "no", "draft_claim": "doubling from 6 to 12 epochs added five points",
           "note": "6-epoch captions and perturbation were overwritten; not traceable"})
paired("§11.2", "strong adaptive v (12 epochs), defended", "nb06/clean, defended", "nb06/strong adaptive v, defended",
       "strong adaptive v (12 epochs), no defense", "nb06/clean", "nb06/strong adaptive v, no defense",
       draft="does better against the defended model (35.5%) than the undefended one (29.5%)",
       note="the two rates use different clean references (defended vs undefended clean captions), as in the draft")
paired("§6 splitting", "real pipeline, splitting off", "nb04/clean", "nb04/v10", "real pipeline, splitting on", "nb06/clean", "nb06/fixed v, no defense",
       draft="splitting reduces the attack by about ten points",
       note="each run is scored against its own clean captions (splitting off vs on)")


# ----------------------------------------------------------------------------------------------
# Step 6
# ----------------------------------------------------------------------------------------------
S6 = {}
_, gc7 = flags("nb06/clean", "nb06/clean, defended", THR_STRICT); r, lo, hi, n = rate(gc7)
S6["global_tile_clean_cost_at_0.7"] = {"value": round(r, 1), "ci": [round(lo, 1), round(hi, 1)], "n": n, "source": "week3_strong_adaptive.json"}
add_value("§11.4 (missing)", "global tile resize-pad: clean cost at 0.7", "--", f"{r:.1f}%", n, "week3_strong_adaptive.json", "verified",
          "new", lo=f"{lo:.1f}", hi=f"{hi:.1f}", note="absent from the draft table; computed from saved captions")
S6["yes_no_breakdown"] = "NOT TRACEABLE: nb02 saved only aggregates (80% flipped, 28% valid) and 3 example answers"
cl, pv = C["all_captions"]["clean"], C["all_captions"]["universal v"]
has_p = lambda c: "person" in c.lower()
has_ws = lambda c: "white shirt" in c.lower()
no_person_clean = [i for i, c in enumerate(cl) if not person3(c)]
S6["attractor"] = {
    "clean_person_rate_nb03_definition": round(100 * np.mean([person3(c) for c in cl]), 1),
    "clean_person_rate_word_person_only": round(100 * np.mean([has_p(c) for c in cl]), 1),
    "perturbed_white_shirt": int(sum(map(has_ws, pv))),
    "perturbed_white_shirt_without_person": int(sum(has_ws(c) and not has_p(c) for c in pv)),
    "perturbed_person_rate_among_images_whose_clean_caption_has_no_person": {
        "n": len(no_person_clean),
        "rate": round(100 * np.mean([person3(pv[i]) for i in no_person_clean]), 1)},
    "definition_note": "nb03 counts 'person' or 'man ' or 'woman'; nb04 (§9) counts 'person' or 'white shirt'",
}
S6["token_cos_0.9_check"] = {"value": round(100 * ev["random ±ξ"]["emb_fool@0.9"], 1), "setting": "after preprocessing, splitting off, random ±ξ (CUDA draw)",
                             "n_images": 500, "per_image": "not saved"}
S6["reeval_reproducibility"] = reeval_check()
_a = reeval_values("A_tokcos_after_preproc.csv", "token_cos", condition="random (CUDA draw)")
if _a is not None:
    S6["token_cos_0.9_check"]["recomputed_from_reeval"] = round(100 * float((_a < 0.9).mean()), 1)
json.dump(S6, open(RES / "step6.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


# ----------------------------------------------------------------------------------------------
# write outputs
# ----------------------------------------------------------------------------------------------
def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
write(ROOT / "all_numbers.csv", ROWS)
write(ROOT / "paired_tests.csv", PT)
write(RES / "similarities.csv", SIM_ROWS)
write(RES / "near_threshold.csv", NEAR or [{"pair": "", "image_id": "", "threshold": "", "similarity": ""}])
print(f"image_id source: {ID_SOURCE}")
print(f"{len(ROWS)} numbers, {sum(r['matches_draft'] == 'no' for r in ROWS)} mismatches, "
      f"{sum(r['matches_draft'] == 'not traceable' for r in ROWS)} not traceable; {len(PT)} paired tests; "
      f"{len(NEAR)} pairs within 0.001 of a threshold")
