# Statistical audit: recap

Draft audited: *Universal Adversarial Perturbations on a Tiny Vision-Language Model* (Nov 12, 2026).
Everything below was recomputed from the saved files in `smol_exp/uap_smolvlm/`, which were not modified, plus per-image outputs from one Colab re-run of `reeval_colab.py` (in `reeval/stats_audit_reeval/`). The re-run loaded only saved perturbations and trained nothing. Caption similarity was recomputed locally on CPU with the same MiniLM checkpoint as the notebooks (`all-MiniLM-L6-v2`).

Files in this folder:
- `all_numbers.csv`: 102 numbers from the draft, each with its recomputed value, Wilson interval, n and source.
- `paired_tests.csv`: 22 exact McNemar comparisons.
- `DRAFT_EDITS.md`: every change to make in the draft, as current text → replacement.
- `PROVENANCE.md`: Step 1, where each number comes from.
- `audit.py`: the full analysis; reruns from scratch.
- `reeval_colab.py`: the GPU re-evaluation, run once on Colab; its outputs are in `reeval/stats_audit_reeval/`.
- `results/`: per-image similarities, Step 6 numbers, LaTeX rows and the image-alignment check.

**Headline.** All 102 numbers from the draft were checked:
- **74 percentages** were recomputed from per-image captions, with a Wilson interval each. 40 come from saved captions, 10 from stand-in captions of an identical run, and 24 from the Colab re-run. 71 match the draft to the printed decimal, 2 are a wording slip and a rounding slip (items 5 and 6 below), and 1 is new (the global tile's clean cost at 0.7, which is missing from the draft).
- **18 means** (token cosines, PSNR) were recomputed per image and all match. They now have SDs (§4 below).
- **3 counts** match.
- **3 means** remain aggregate-only and match their saved values: the GoogLeNet token cosine (the optional GoogLeNet file was not uploaded) and the two §11.3 training-log values.
- **4 numbers are NOT TRACEABLE:** the 6-epoch 30.5%, and 3 §10 numbers that need the human labels.

The statistics change two of the draft's conclusions:
- "there is no single best defense";
- "the global-tile defense is the cheapest on clean captions".

They also weaken a third: that the 12-epoch perturbation does better against the defended model than the undefended one.

### How images were matched

No caption file stores an image ID. I rebuilt the notebooks' exact file list from the ImageNetV2 archive (`build_file_list.py`), so every comparison is merged on the real file name (`data/heldout_files.json`). Four checks confirm that position i is the same image in every file (`alignment_check.py`, `results/alignment_check.json`):
1. Identical captions across nb03, nb04 and nb07: 200/200.
2. The few examples notebook 02 saved sit at the expected positions: 3/3 in each setting.
3. Across settings, caption similarity peaks at shift 0 and never at any other shift.
4. Against the true ImageNet class of each file, 33% of clean captions rank their true class in the top 10 of 1,000 at the same position, against about 1% (chance) at every shifted position.

The 50- and 100-image runs are always the first 50 or 100 of the same 200 images.

---

## 1. Discrepancies between the draft and the data (most important first)

1. **§11.2: the 6-epoch result and the training log cannot be traced.**
   - The 30.5% for the 6-epoch strong attack exists only as a printed number. Its captions and its perturbation were overwritten by the 12-epoch rerun, whose checkpoint stores `epoch = 12`.
   - The training log quoted in §11.2 (local-tile cosine 0.358 → 0.251, defended global tile 0.676 → 0.634) appears in no file. The log saved in notebook 06 is from a different run: epochs 7–10 of 10, with 0.272 → 0.261 and 0.645 → 0.641.
   - Notebook 06's code now says `STRONG_EPOCHS = 6`, so it doesn't reproduce the 12-epoch result as written.
   - Either drop the 6-epoch row and the cosine log, or label them as values from an earlier, unsaved run.
2. **The token cosine image counts are wrong or missing in several places:**
   - The 0.142 in the §6 and §8 tables is computed on the first **50** images, while the tables say 200.
   - The §8 values for 500M (0.706) and 2.2B (0.871) use **30** images.
   - The §9 cosines and PSNR use 50 images.
   - The §11.3 "0.675 vs 0.361" is the attacker's **training loss on the 500 build images**, not a measurement on held-out images.
   - The §11.3 drift values (0.971 and 0.953) use 50 images.
   
   State n for each. For §11.3, either say "on the build images during training" or re-measure on held-out images.
3. **Two different random-noise draws are mixed (§6 and §8 vs §9).** Notebooks 01 and 02 drew the noise on the GPU; notebooks 03, 04 and 05 drew it on the CPU. The same seed gives different noise on each device.
   - Through the real pipeline with splitting off, the GPU draw gives 3.5% (§6, §8) and the CPU draw 4.0% (§9, abstract).
   - Both are correct. The re-run reproduces the GPU-draw captions and gives exactly 3.5% (7/200).
   - Use one draw in all three tables, or add a footnote explaining the two draws.
4. **Fig. 12 is stale.** It shows 28% for random resize-and-pad, which is the epoch-7 value. The table and the saved captions are from epoch 8 (29.0%). Redraw it.
5. **§6, "white shirt in roughly one perturbed caption in six":** the data give 27/200 = 13.5% (interval 9.4–18.9%), closer to one in seven. All 27 of those captions also mention a person.
6. **§9, "person or white shirt" at 16/255:** the value is 18.5%, printed as "18%" by Python's round-half-to-even. Write 18.5% or 19%.
7. **Two different "person" definitions:**
   - §6 (14% → 28%) counts "person", "man" or "woman".
   - The §9 column counts "person" or "white shirt", which gives 11% for clean captions, not 14%.
   - The 28% at 10/255 is the same under both, by coincidence (27.5% each).
   - Say which definition each section uses.
8. **These numbers looked suspicious but are consistent:**
   - **5.0% (200 images) vs 2% (100 images), random noise after preprocessing:** same noise draw. The first 100 images give exactly 2/100, so 3 of the 5 noise-induced changes fall in images 101–200.
   - **70.0% / 59.0% / 60.5% with splitting on:** nested samples of the same run. The first 50, 100 and 200 images give exactly those values. The first 50 simply landed high.
   - **Global tile, 1.0 / 2.0 / 10.0% (100 images, §11.1) vs 2.5 / 1.5 / 13.0% (200 images, §11.2):** the first 100 images of notebook 06 give exactly 1.0 / 2.0 / 10.0%, so the gap comes only from images 101–200.
   - **The "50 vs 5" claim:** confirmed for the 12-epoch perturbation (50 vs 5). The 38 vs 3 printed in notebook 06 belongs to the lost 6-epoch run.
   - **§10 counts:** confirmed. The budget sweep has exactly 377 exact-match pairs, and the 60-pair key has 37 universal / 23 random pairs and 10 pairs at 16/255.
9. **Precision:**
   - Empty captions occur 1–3 times per perturbed run. They score 0.08–0.16, so each one counts as changed. This is reasonable; say so in §4.4.
   - 30 caption pairs fall within 0.001 of a threshold (`results/near_threshold.csv`, now including the re-run captions). The closest is 2×10⁻⁵ from 0.5, which is well above the CPU-vs-GPU difference (about 10⁻⁶). In any case, every rate reproduced exactly.

---

## 2. Which comparisons are statistically supported

Every percentage has a Wilson 95% interval in `all_numbers.csv`. Paired comparisons use the exact McNemar test on images shared by both conditions, merged on `image_id`. All of them have 200 paired images.

### Attack vs random noise (Wilson intervals only)

The intervals are far apart in every setting: §5 (all prompts), §6, §8 for 256M and 500M, and §9 (all budgets). The one exception is **2.2B**: 7.0% [3.4, 13.7] against 3.0% [1.0, 8.5]. Those intervals overlap, which supports the draft's own wording, "essentially unaffected".

### Comparisons where the draft relies on a close gap

| Comparison | A | B | only A / only B | Difference | p | Does it support the draft? |
|---|---|---|---|---|---|---|
| Splitting off vs on (§6) | 71.5% | 60.5% | 28 / 6 | +11.0 | 0.0002 | **Yes**, "about ten points" |
| Strong vs weak adaptive (§11.2) | 35.5% | 13.0% | 50 / 5 | +22.5 | 2×10⁻¹⁰ | **Yes** |
| 6 vs 12 epochs (§11.2) | 30.5% | 35.5% | — | — | not computable | **Not testable**: the 6-epoch captions are lost |
| 12-epoch v, defended vs undefended (§11.2) | 35.5% | 29.5% | 26 / 14 | +6.0 | 0.081 | **No**: not significant |

**§11.2, "does better against the defended model (35.5%) than against the undefended one (29.5%) … an attacker has to commit to one of the two":** the paired test doesn't support this (p = 0.08). Suggested wording: "the 12-epoch perturbation is about as effective against the undefended model (29.5%) as against the defended one (35.5%; McNemar p = 0.08)." Drop "has to commit".

### §11.4 ranking (random resize-and-pad, FT-UAT, FT-fixed, all with splitting off)

| Comparison | A | B | only A / only B | Difference | p |
|---|---|---|---|---|---|
| Strongest attack: random resize-pad vs FT-UAT | 29.0% | 37.5% | 18 / 35 | −8.5 | **0.027** |
| Strongest attack: random resize-pad vs FT-fixed | 29.0% | 41.0% | 18 / 42 | −12.0 | **0.003** |
| Strongest attack: FT-UAT vs FT-fixed | 37.5% | 41.0% | 21 / 28 | −3.5 | 0.39 |
| Clean cost at 0.5: random resize-pad vs FT-UAT | 6.0% | 4.5% | 7 / 4 | +1.5 | 0.55 |
| Clean cost at 0.5: random resize-pad vs FT-fixed | 6.0% | 3.0% | 9 / 3 | +3.0 | 0.15 |
| Clean cost at 0.5: FT-UAT vs FT-fixed | 4.5% | 3.0% | 5 / 2 | +1.5 | 0.45 |
| Clean cost at 0.7: random resize-pad vs FT-UAT | 25.0% | 13.5% | 28 / 5 | +11.5 | **7×10⁻⁵** |
| Clean cost at 0.7: random resize-pad vs FT-fixed | 25.0% | 8.0% | 38 / 4 | +17.0 | **6×10⁻⁸** |
| Clean cost at 0.7: FT-UAT vs FT-fixed | 13.5% | 8.0% | 14 / 3 | +5.5 | **0.013** |

**"There is no single best defense" should change.** The draft bases it on overlapping intervals, but the paired test says otherwise:
- **Attack rate:** random resize-and-pad leaves significantly less residual attack than either fine-tuned encoder (p = 0.027 and p = 0.003). The two fine-tuned encoders don't differ from each other.
- **Clean cost:** the ordering reverses. At 0.7, random resize-and-pad alters significantly more normal captions than both fine-tuned encoders. At 0.5, no defense differs from another.

So there is a best defense on each criterion, but not on both. Suggested wording:

> "Random resize-and-pad leaves the least residual attack (29.0%, significantly less than FT-UAT and FT-fixed; McNemar p = 0.027 and 0.003), but it also changes significantly more clean captions at the stricter threshold (25.0% vs 13.5% and 8.0%; p < 0.001). No defense is best on both counts."

Two caveats belong next to it:
- Each defense was tested against a different attacker: BPDA+EOT for 8 epochs on random resize-and-pad, a plain adaptive UAP for 7 epochs on the fine-tuned encoders. The comparison is between the *strongest tested* attacks, as the draft says.
- p = 0.027 is a modest effect for a set of three comparisons.

### Global-tile defense (confounded: it runs with splitting on, the others with splitting off)

| Comparison (global tile = A) | A | B | only A / only B | p |
|---|---|---|---|---|
| Strongest attack vs random resize-pad | 35.5% | 29.0% | 29 / 16 | 0.072 |
| Strongest attack vs FT-UAT | 35.5% | 37.5% | 25 / 29 | 0.68 |
| Strongest attack vs FT-fixed | 35.5% | 41.0% | 22 / 33 | 0.18 |
| Clean cost at 0.5 vs random resize-pad / FT-UAT / FT-fixed | 2.5% | 6.0 / 4.5 / 3.0% | 4/11, 4/8, 5/6 | 0.12 / 0.39 / 1.0 |
| Clean cost at 0.7 vs random resize-pad / FT-UAT / FT-fixed | **15.5%** | 25.0 / 13.5 / 8.0% | 14/33, 23/19, 28/13 | 0.008 / 0.64 / **0.028** |

**"The global-tile defense is the cheapest on clean captions" is not supported.**
- At 0.5, its 2.5% doesn't differ from any other defense (all p ≥ 0.12).
- At 0.7, its clean cost (15.5% [11.1, 21.2], the value missing from the draft's table) is *higher* than FT-fixed's (8.0%, p = 0.028), similar to FT-UAT's, and lower only than random resize-and-pad's.
- All of this is confounded by splitting on vs off.

Suggested wording:

> "The global-tile defense costs about as little on clean captions as the fine-tuned encoders (2.5% at the main threshold), and it needs no training, but at the stricter threshold it alters more captions than FT-fixed (15.5% vs 8.0%)."

Its residual attack (35.5%) doesn't differ significantly from any other defense.

---

## 3. Human check (§10): cleaned up as far as the data allow

You asked to set this part aside. The human labels aren't in the repository, so these items can't be checked yet: the condition × label table, the 44 decided pairs, the 0.755 / 0.652 boundary, κ, and the six misses.

**Confirmed from saved data:**
- The 1,600 candidate pairs come from the notebook 04 budget sweep.
- 377 exact string matches were removed.
- The 60 labelled pairs come one per image (60 distinct images, positions 9–194), split 37 universal / 23 random, with 10 universal pairs at 16/255.
- The code that drew the 60 pairs is not in the repository, so the sampling design (strata, counts per budget) cannot be confirmed. The key's `lexical_ratio` column matches a plain `difflib` ratio for 48 of the 60 pairs, so some variant of that ratio was probably used to oversample borderline cases.

**The draft's own counts with Wilson intervals,** for use once the labels confirm them:

| Quantity | Count | Rate [95% Wilson] |
|---|---|---|
| Universal pairs judged changed, of all 37 | 27/37 | 73.0% [57.0, 84.6] |
| Universal pairs judged changed, of the 30 decided | 27/30 | 90.0% [74.4, 96.5] |
| Random pairs judged changed, of all 23 | 3/23 | 13.0% [4.5, 32.1] |
| Random pairs judged changed, of the 14 decided | 3/14 | 21.4% [7.6, 47.6] |
| Agreement at 0.3 / 0.5 / 0.6 / 0.7 (44 decided pairs) | 31 / 38 / 43 / 44 | 70.5% [55.8, 81.8] / 86.4% [73.3, 93.6] / 97.7% [88.2, 99.6] / 100% [92.0, 100] |

Report κ as "κ = 0.72 (one annotator, 44 decided pairs)", without an interval. To finish this section, I need the labels file: pair ID → changed / unchanged / unsure. `audit.py` would then rebuild the table, κ, the misses (condition, budget and similarity of each) and the similarities of the 16 unsure pairs.

---

## 4. Per-image re-evaluation (Colab): done

`reeval_colab.py` ran parts A–F on Colab. It loaded only saved perturbations, trained nothing, and keyed every row by `image_id`. Its outputs are in `reeval/stats_audit_reeval/`, and `audit.py` reads them directly.

**The re-run reproduces the original runs exactly.** Wherever it overlaps a saved caption file, every caption is identical, string for string (`results/step6.json`, `reeval_reproducibility`):

| Re-run captions | Saved captions | Identical |
|---|---|---|
| nb02 "describe": clean, universal v, GPU-drawn noise | nb01 | 100/100 each |
| nb02 real pipeline, splitting off, clean | nb04 | 200/200 |
| nb02 real pipeline, splitting on, clean | nb06 | 50/50 |
| nb05 undefended clean, splitting off | nb04 | 200/200 |
| nb05 undefended clean, splitting on | nb06 | 100/100 |

All 24 rates recomputed from re-run captions match the draft to the printed decimal, as do the 18 per-image means:
- §5 prompt table, including the yes/no flip (80% / 3%) and still-valid rates (28% / 100%);
- the GPU-draw noise through the real pipeline (3.5%, 2.0%);
- §8 transfer (500M and 2.2B);
- §11.1 JPEG, blur and blanked global tile;
- §12 "91.6% of random noise passes token cosine < 0.9" (now 91.6% [88.8, 93.7], 500 images).

**Means with SD and n,** for the tables. The n is the number of images each mean was actually computed on; several differ from what the draft states (item 2 above).

| Draft | Quantity | Mean | SD | n |
|---|---|---|---|---|
| §5 | token cosine after preprocessing, universal v | −0.051 | 0.256 | 500 |
| §5 | token cosine after preprocessing, random noise | 0.846 | 0.041 | 500 |
| §5 | token cosine, GoogLeNet UAP | 0.850 | not available | 500 |
| §6, §8, §9 | token cosine, real pipeline off, universal v (10/255) | 0.142 | 0.342 | 50 |
| §6 | token cosine, splitting on, mean over 17 tiles | 0.688 | 0.051 | 50 |
| §7 | local tiles, universal v / random noise | 0.723 / 0.771 | 0.038 / 0.021 | 50 |
| §7 | global tile, universal v / random noise | 0.142 / 0.865 | 0.342 / 0.038 | 50 |
| §8 | token cosine, 500M / 2.2B | 0.706 / 0.871 | 0.101 / 0.039 | 30 |
| §9 | token cosine at 4 / 8 / 10 / 16 per 255 | 0.623 / 0.291 / 0.142 / −0.054 | 0.152 / 0.355 / 0.342 / 0.283 | 50 |
| §9 | PSNR (dB) at 4 / 8 / 10 / 16 per 255 | 37.05 / 31.98 / 30.08 / 26.43 | 0.26 / 0.28 / 0.30 / 0.33 | 50 |

Two points for the text:
- The global-tile cosine with splitting on (§7) and the splitting-off cosine (§6) are identical image by image (to within 3×10⁻⁶ on all 50 images), 0.142 with SD 0.342. With splitting off, the single tile *is* the global tile.
- The attack's token cosines vary much more across images (SD 0.26–0.36) than random noise does (SD 0.02–0.04). At 8/255 and above, the mean hides a spread from images that are almost untouched to images that are fully disrupted.

**Still missing (optional):** the GoogLeNet SD needs `universal_googlenet_linf.npy` in `MyDrive/uap_replication/` and part A run again (about 3 minutes).

**Not recoverable without retraining (out of scope):**
- the 6-epoch strong perturbation;
- the §11.2 and §11.4 training logs (local/global cosines, losses 0.543 → 0.541, the epoch-7 rate of 28.0%);
- the FT training logs.

**Optional improvement:** the §11.3 "0.675 vs 0.361" could be re-measured on held-out images using the saved adaptive perturbations and the fine-tuned encoders on your Drive (`encoder_FT-*.pt`). Those encoders aren't in this repository, so I haven't added that part.

---

## 5. Ready to paste

### LaTeX rows

Rate with a 95% Wilson interval, every one computed from per-image captions except the † row, which exists only as a printed number (its interval comes from k = rate × n). The full set is in `results/latex_rows.tex`.

```latex
% Section 6
After preprocessing & 200 & 1 & 91.0\% \scriptsize[86.2, 94.2] & 5.0\% \scriptsize[2.7, 9.0] \\
Real pipeline, splitting off & 200 & 1 & 71.5\% \scriptsize[64.9, 77.3] & 3.5\% \scriptsize[1.7, 7.0] \\
Real pipeline, splitting on & 50 & 17 & 70.0\% \scriptsize[56.2, 80.9] & 2.0\% \scriptsize[0.4, 10.5] \\
Real pipeline, splitting on & 200 & 17 & 60.5\% \scriptsize[53.6, 67.0] & -- \\

% Section 8
256M & 200 & 71.5\% \scriptsize[64.9, 77.3] & 3.5\% \scriptsize[1.7, 7.0] \\
500M & 200 & 31.0\% \scriptsize[25.0, 37.7] & 10.0\% \scriptsize[6.6, 14.9] \\
2.2B & 100 & 7.0\% \scriptsize[3.4, 13.7] & 3.0\% \scriptsize[1.0, 8.5] \\

% Section 9
4/255 & 21.5\% \scriptsize[16.4, 27.7] & 1.0\% \scriptsize[0.3, 3.6] \\
8/255 & 62.5\% \scriptsize[55.6, 68.9] & 3.5\% \scriptsize[1.7, 7.0] \\
10/255 & 71.5\% \scriptsize[64.9, 77.3] & 4.0\% \scriptsize[2.0, 7.7] \\
16/255 & 91.5\% \scriptsize[86.8, 94.6] & 4.0\% \scriptsize[2.0, 7.7] \\

% Section 11.1 (clean cost / fixed v / weak adaptive v)
JPEG, quality 75 & 200 & 3.0\% \scriptsize[1.4, 6.4] & 8.5\% \scriptsize[5.4, 13.2] & 23.0\% \scriptsize[17.7, 29.3] \\
Gaussian blur, $\sigma=1$ & 200 & 6.0\% \scriptsize[3.5, 10.2] & 5.0\% \scriptsize[2.7, 9.0] & 29.5\% \scriptsize[23.6, 36.2] \\
Random resize-and-pad & 200 & 6.0\% \scriptsize[3.5, 10.2] & 4.0\% \scriptsize[2.0, 7.7] & 21.0\% \scriptsize[15.9, 27.2] \\
Global tile: resize-and-pad & 100 & 1.0\% \scriptsize[0.2, 5.4] & 2.0\% \scriptsize[0.6, 7.0] & 10.0\% \scriptsize[5.5, 17.4] \\
Global tile: blanked & 100 & 23.0\% \scriptsize[15.8, 32.2] & 20.0\% \scriptsize[13.3, 28.9] & 80.0\% \scriptsize[71.1, 86.7] \\

% Section 11.2 (global-tile defense, splitting on, 200 images)
Clean cost of the defense & 2.5\% \scriptsize[1.1, 5.7] \\
Fixed $v$, no defense & 60.5\% \scriptsize[53.6, 67.0] \\
Fixed $v$, defended & 1.5\% \scriptsize[0.5, 4.3] \\
Weak adaptive $v$, defended & 13.0\% \scriptsize[9.0, 18.4] \\
Strong adaptive $v$, defended, 6 epochs & 30.5\% \scriptsize[24.5, 37.2]$^\dagger$ \\
Strong adaptive $v$, defended, 12 epochs & 35.5\% \scriptsize[29.2, 42.3] \\
Strong adaptive $v$ (12 epochs), no defense & 29.5\% \scriptsize[23.6, 36.2] \\

% Section 11.4 (clean cost at 0.5 / 0.7, strongest adaptive attack)
Random resize-and-pad & splitting off & 71.5\% & 6.0\% \scriptsize[3.5, 10.2] / 25.0\% \scriptsize[19.5, 31.4] & 29.0\% \scriptsize[23.2, 35.6] \\
Global tile: resize-and-pad & splitting on & 60.5\% & 2.5\% \scriptsize[1.1, 5.7] / 15.5\% \scriptsize[11.1, 21.2] & 35.5\% \scriptsize[29.2, 42.3] \\
FT-UAT & splitting off & 71.5\% & 4.5\% \scriptsize[2.4, 8.3] / 13.5\% \scriptsize[9.4, 18.9] & 37.5\% \scriptsize[31.1, 44.4] \\
FT-fixed & splitting off & 71.5\% & 3.0\% \scriptsize[1.4, 6.4] / 8.0\% \scriptsize[5.0, 12.6] & 41.0\% \scriptsize[34.4, 47.9] \\
```

The intervals the draft already reports in §11.2 and §11.4 all match these to the printed decimal.

### Sentence for Section 4.5

> Every rate is reported with a 95% Wilson interval; when two conditions are evaluated on the same images, we compare them with an exact McNemar test on the images whose outcome differs between the two conditions, and we measure agreement between the metric and the human annotator with Cohen's κ. These intervals and tests reflect only which held-out images were evaluated: each perturbation and each fine-tuned encoder was trained once, with a single seed, so they do not capture variation from training randomness.
