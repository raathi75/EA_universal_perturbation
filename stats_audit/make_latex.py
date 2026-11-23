"""Ready-to-paste LaTeX rows (rate with 95% Wilson interval) built from all_numbers.csv."""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parent
with open(ROOT / "all_numbers.csv", encoding="utf-8") as fh:
    R = {r["description"]: r for r in csv.DictReader(fh)}
def c(desc, plain=False):
    r = R[desc]; v = r["recomputed_value"].split("%")[0].split(" ")[0]
    if r["ci_low"] == "": return v
    s = f"{float(v):.1f}" + r"\% \scriptsize" + f"[{r['ci_low']}, {r['ci_high']}]"
    return s + (r"$^\dagger$" if r["matches_draft"] == "not traceable" else "")
L = []
L.append("% Section 6: settings (dagger = saved aggregate only, interval from k = rate x n)")
L.append(f"After preprocessing & 200 & 1 & {c('universal v, after preprocessing, captions changed')} & {c('random ±ξ (CUDA draw), after preprocessing')} \\\\")
L.append(f"Real pipeline, splitting off & 200 & 1 & {c('universal v, real pipeline, splitting off')} & {c('random ±ξ (CUDA draw, nb02), real pipeline, splitting off')} \\\\")
L.append(f"Real pipeline, splitting on & 50 & 17 & {c('splitting on, first 50 images (stand-in: nb06 captions)')} & {c('random ±ξ, splitting on, 50 images')} \\\\")
L.append(f"Real pipeline, splitting on & 200 & 17 & {c('universal v, real pipeline, splitting on (121/200)')} & -- \\\\")
L.append("")
L.append("% Section 8: transfer (real pipeline, splitting off)")
L.append(f"256M & 200 & {c('universal v, real pipeline, splitting off')} & {c('random ±ξ (CUDA draw, nb02), real pipeline, splitting off')} \\\\")
L.append(f"500M & 200 & {c('500M, universal v')} & {c('500M, random ±ξ')} \\\\")
L.append(f"2.2B & 100 & {c('2.2B, universal v')} & {c('2.2B, random ±ξ')} \\\\")
L.append("")
L.append("% Section 9: budget sweep (200 images, real pipeline, splitting off)")
for e in (4, 8, 10, 16):
    v = c("universal v, real pipeline, splitting off") if e == 10 else c(f"universal v, {e}/255")
    rd = c("random ±ξ (CPU draw), real pipeline, splitting off, 10/255") if e == 10 else c(f"random ±ξ, {e}/255")
    L.append(f"{e}/255 & {v} & {rd} \\\\")
L.append("")
L.append("% Section 11.1: test-time defenses (clean cost / fixed v / weak adaptive v)")
for name, n, keys in [("JPEG, quality 75", 200, ("JPEG q75: clean cost", "JPEG q75: fixed v", "JPEG q75: weak adaptive v")),
                      (r"Gaussian blur, $\sigma=1$", 200, ("blur σ1: clean cost", "blur σ1: fixed v", "blur σ1: weak adaptive v")),
                      ("Random resize-and-pad", 200, ("random resize-pad: clean cost (stand-in: nb07)", "random resize-pad: fixed v (stand-in: nb07)", "random resize-pad: weak adaptive v (stand-in: nb07)")),
                      ("Global tile: resize-and-pad", 100, ("global tile resize-pad: clean cost, 100 images (stand-in: nb06 first 100)", "global tile resize-pad: fixed v, 100 images (stand-in: nb06 first 100)", "global tile resize-pad: weak adaptive v, 100 images (stand-in: nb06 first 100)")),
                      ("Global tile: blanked", 100, ("global tile: blanked: clean cost", "global tile: blanked: fixed v", "global tile: blanked: weak adaptive v"))]:
    L.append(f"{name} & {n} & " + " & ".join(c(k) for k in keys) + " \\\\")
L.append("")
L.append("% Section 11.2: global-tile defense, splitting on, 200 images")
for lab, k in [("Clean cost of the defense", "global tile: clean cost (5/200)"), ("Fixed $v$, no defense", "universal v, real pipeline, splitting on (121/200)"),
               ("Fixed $v$, defended", "global tile: fixed v, defended"), ("Weak adaptive $v$, defended", "global tile: weak adaptive v, defended"),
               ("Strong adaptive $v$, defended, 6 epochs", "global tile: strong adaptive v, defended, after 6 epochs"),
               ("Strong adaptive $v$, defended, 12 epochs", "global tile: strong adaptive v (12 epochs), defended"),
               ("Strong adaptive $v$ (12 epochs), no defense", "strong adaptive v (12 epochs), no defense")]:
    L.append(f"{lab} & {c(k)} \\\\")
L.append("")
L.append("% Section 11.4: ranking (clean cost at 0.5 / 0.7, strongest adaptive attack)")
L.append(f"Random resize-and-pad & splitting off & 71.5\\% & {c('random resize-pad: clean cost at 0.5')} / {c('random resize-pad: clean cost at 0.7')} & {c('random resize-pad: strong adaptive v (8 epochs)')} \\\\")
L.append(f"Global tile: resize-and-pad & splitting on & 60.5\\% & {c('global tile: clean cost (5/200)')} / {c('global tile resize-pad: clean cost at 0.7')} & {c('global tile: strong adaptive v (12 epochs), defended')} \\\\")
L.append(f"FT-UAT & splitting off & 71.5\\% & {c('FT-UAT: clean cost at 0.5')} / {c('FT-UAT: clean cost at 0.7')} & {c('FT-UAT: adaptive v')} \\\\")
L.append(f"FT-fixed & splitting off & 71.5\\% & {c('FT-fixed: clean cost at 0.5')} / {c('FT-fixed: clean cost at 0.7')} & {c('FT-fixed: adaptive v')} \\\\")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "latex_rows.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
