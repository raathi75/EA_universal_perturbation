# Step 1: Inventory and provenance

Draft audited: `Universal_Adversarial_Perturbations_on_a_Tiny_Vision_Language_Model.pdf` (Nov 12, 2026).
Saved data: `smol_exp/uap_smolvlm/`. Code: `smol_exp/01…07_*.ipynb`. No file was modified.

## A. What is saved, and what is not

### Per-image captions (the only per-image data that exist)

| File | Written by | Caption lists (each 200 long, unless noted) | Setting |
|---|---|---|---|
| `week1_results.json` | nb01 | clean, universal v, random ±ξ, replication v (GoogLeNet) | After preprocessing, splitting off, ε=10 |
| `week2_step2_results.json` → `C_dominant/all_captions` | nb03 part C | clean, universal v, random ±ξ | Real pipeline, splitting off, ε=10 |
| `week2_step2_results.json` → `E_short` | nb03 part E | clean_answers, universal v, random ±ξ (100 each) | Real pipeline, off, "main object" prompt |
| `week2_step3_eps_sweep.json` → `captions` | nb04 | clean + {universal v, random ±ξ} × ε {4, 8, 10, 16} | Real pipeline, splitting off |
| `week3_strong_adaptive.json` → `captions` | nb06 | clean; clean, defended; fixed v (no defense / defended); weak adaptive v, defended; strong adaptive v (defended / no defense) | Real pipeline, **splitting on**, global-tile rrp |
| `week3_finetune_and_rrp.json` → `captions` | nb07 | clean, original encoder; clean, rrp defense; fixed v (no defense / rrp); weak adaptive v, rrp; strong adaptive v, rrp; and clean / fixed v / adaptive v for FT-fixed and for FT-UAT | Real pipeline, splitting off |

### Missing entirely

- **No per-image similarity scores.** All rates must be recomputed from the captions with MiniLM; see question 1.
- **No per-image token cosine anywhere.** Every token cosine in the draft is a saved mean only, so SDs, bootstraps and Wilcoxon tests need re-evaluation on Colab.
- **No image IDs or filenames** are saved with any caption list. See section B.
- **Captions not saved:**
  - nb02: prompt table, B1, B2. Only 3 examples per condition are kept.
  - nb03 part F: 500M and 2.2B transfer. Only 4 examples are kept.
  - nb05: every row of §11.1.
  - nb06: the 6-epoch strong v. Its captions were overwritten by the 12-epoch rerun.
- **Section 10:** no human labels and no sampling code. Only `labeling_key_DO_NOT_OPEN_before_labeling.csv` exists (pair, image_index, condition, lexical_ratio).
- **Checkpoints:** `strong_adaptive_globalrrp_ckpt.pt` stores `{v, epoch=12, opt}` and `strong_adaptive_rrp_ckpt.pt` stores `{v, epoch=8, opt}`. Neither has loss or cosine logs, and `uap_ckpt.pt` holds v only.

### Code facts

- **Caption metric:** `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` with normalised embeddings. The similarity is a dot product, and a caption is "changed" if it is `< 0.5` (strict). This is the same in all 7 notebooks.
- **Empty captions:** no special case. `""` is embedded like any other string, and generated captions are `.strip()`-ed first. Empty captions occur 1× in nb01 v, 1× in nb03/04/07 v ε=10, 1× in ε=16, 3× in nb06 fixed v (splitting on) and 1× in nb03-E v.
- **Image order:**
  - Every notebook runs `FILES = sorted(glob(".../*/*.jpeg"))`, then `perm = RandomState(0).permutation(10000)`. Images `perm[:500]` build v, and the held-out images are `perm[500:1000]`.
  - Every n-image run uses `E_u8[:n]`, so the 50- and 100-image runs are the first 50/100 of the same 200.
  - Evidence: the clean captions are identical image by image across nb03, nb04 and nb07 (200/200), and so are the universal-v ε=10 captions (200/200).
- **Random-noise draws:**
  - Every notebook calls `torch.manual_seed(0)` and then `sign(randn_like(v))`.
  - In nb01 and nb02, v lives on the GPU, so the noise is drawn by the CUDA RNG.
  - In nb03, nb04 and nb05, v is on the CPU, so the noise is drawn by the CPU RNG, which gives a different pattern from the same seed.
  - The nb03 and nb04 random captions are identical (200/200), which confirms they share one draw. nb02 (3.5%) used another draw.
- **Token cosine sample sizes the draft doesn't state:**
  - nb02 computes B1/B2 cosines on the first 50 images.
  - nb03-F computes 500M/2.2B cosines on the first 30.
  - nb04 computes cosines and PSNR on 50.
  - nb01 computes cosine on 500.
  - The FT drift figure uses 50 images.
  - The FT "attacker token cosine" (0.675 / 0.361) is a training-loss value on the build set, not held-out data.
- **Two different "person" definitions:**
  - nb03-C counts `"person" or "man " or "woman"`: 13.5% clean, 27.5% under v.
  - nb04 counts `"person" or "white shirt"`: 11% clean, 27.5% under v at ε=10.
  - §6 uses the first definition and §9 uses the second.
- **Section 10 key:**
  - It points at the nb04 captions: 60 distinct image indices in 9–194, with conditions matching the 37 universal / 23 random split.
  - Its `lexical_ratio` column matches `difflib.SequenceMatcher` on the raw nb04 captions for 48/60 pairs. The other 12 differ by 0.01–0.05, so it was some variant of that measure.
  - The 377 exact string matches among the 1,600 nb04 pairs are confirmed.

Status codes used below:
- **CAPS**: per-image captions are saved, so the number can be recomputed once similarities are scored.
- **AGG**: only an aggregate exists (a JSON scalar or a printed notebook output), so the number is not traceable per image.
- **NONE**: no saved trace at all.

## B. Provenance table

| # | Section | Draft number | Source | n | Per-image | Note |
|---|---|---|---|---|---|---|
| 1 | Abstract, §2.2, §6, §8, §9 | 71.5% changed (real, off, ε=10) | nb04 `universal v eps10` vs `clean` (identical in nb03-C and nb07 `fixed v, no defense`); aggregate also in nb02 B1 and nb05 | 200 | CAPS | |
| 2 | Abstract, §2.2, §4.5, §6, §11.2 | 60.5% (121/200), splitting on | `week3_strong_adaptive.json` `clean` vs `fixed v, no defense` | 200 | CAPS | |
| 3 | Abstract | "at most 4%" random | nb04 / nb05 CPU-noise draw (4.0%) | 200 | CAPS | Different draw from row 13 |
| 4 | Abstract, §11.4, §15 | 29%–41% strongest attacker | rows 52, 55, 61, 62 | 200 | CAPS | |
| 5 | Abstract, §10 | 86% agreement; "never the reverse" | human labels | 44 | NONE | Labels missing |
| 6 | §4.1 | 500 build / 500 held out / 200 captioned | code (`perm`) | | | Confirmed |
| 7 | §5 table 1 | v: cos −0.051, 91.0% | `week1_results.json` eval (cos); captions `universal v` vs `clean` | 500 cos / 200 cap | cos AGG, rate CAPS | |
| 8 | §5 table 1 | random: cos 0.846, 5.0% | week1 | 500 / 200 | AGG / CAPS | CUDA draw |
| 9 | §5 table 1 | GoogLeNet v: 0.850, 4.0% | week1 | 500 / 200 | AGG / CAPS | |
| 10 | §5 text | "9% survived", "5% under random" | = rows 7, 8 | 200 | CAPS | |
| 11 | §5 prompt table | describe 91% / 2% | nb02 part A output; `week2_step1_results.json` | 100 | AGG | nb02 captions not saved. A proxy is week1's first 100 images (same path, same CUDA noise); it differs from 5.0% only by sample |
| 12 | §5 prompt table | what-is-in 96/12; main-object 94/28; yes/no flip 80/3; valid yes/no 28%/100% | nb02 part A | 100 | AGG | Breakdown of the 80% not traceable |
| 13 | §6, §8 | random 3.5% (real, off) | nb02 B1 | 200 | AGG | CUDA draw, captions not saved |
| 14 | §6 | token cos 0.142 / random 0.864 (off) | nb02 B1 (and nb04 0.142 / 0.865) | **50** | AGG | Draft table says "200 images" |
| 15 | §6 | splitting on, 50 imgs: 70.0% / 2.0% / cos 0.688 (random 0.775) | nb02 B2 | 50 | AGG | First 50 of the nb06 captions can be used as a proxy |
| 16 | §6, §11.1 | 59.0% on 100, splitting on (random 0.0%; 8/255 47.0%) | nb05 `none (splitting on)` | 100 | AGG | First 100 of the nb06 captions as a proxy |
| 17 | §6 | "about twenty points" (91 → 71.5) | rows 7, 1 | 200 | CAPS | |
| 18 | §6 | person mentions 14% → 28%; 173/200 distinct (clean 189, random 179) | nb03-C `mentions_person`, `summary` | 200 | CAPS | Uses the person/man/woman definition |
| 19 | §6 | "white shirt" in ~1 in 6 perturbed captions | not in any JSON | 200 | CAPS | Computable from nb03-C captions |
| 20 | §7 table | local 0.723 (0.68–0.75), global 0.142; random 0.771 (0.75–0.78), 0.865 | nb03-D `mean_cos_per_tile` (17 tile means) | 50 | AGG | No per-image tile cosines |
| 21 | §8 table | 500M: 31.0% / 10.0% / cos 0.706 | nb03-F | 200 rate, **30** cos | AGG | Captions not saved |
| 22 | §8 table | 2.2B: 7.0% / 3.0% / cos 0.871 | nb03-F | 100 rate, **30** cos | AGG | |
| 23 | §8 | 197 tensors, 0 identical, median 1.9% | nb03 cell 17 output | | AGG | Weights only, no stats needed |
| 24 | §9 table | 4/255: 21.5 / 1.0 / cos 0.623 / 14% / PSNR 37.0 | nb04 rows + captions | 200 (cos, PSNR on 50) | rate CAPS, cos AGG | |
| 25 | §9 table | 8/255: 62.5 / 3.5 / 0.291 / 22% / 32.0 | nb04 | same | same | |
| 26 | §9 table | 10/255: 71.5 / 4.0 / 0.142 / 28% / 30.1 | nb04 | same | same | "28%" is 27.5% under the person/white-shirt definition |
| 27 | §9 table | 16/255: 91.5 / 4.0 / −0.054 / 18% / 26.4 | nb04 | same | same | 18.5% → "18%" (rounding) |
| 28 | §9 | flattest region, "30 held-out images" | nb06 cell 16 | 30 | n/a | Visual only |
| 29 | §10 | 1,600 pairs, 377 removed | nb04 captions | 1,600 | CAPS | Confirmed: 377 exact string matches |
| 30 | §10 | 60 pairs, 37 universal / 23 random; 10 at 16/255 | label key | 60 | CAPS | Key confirms 37 / 23 and 10 at ε=16. Sampling code missing |
| 31 | §10 table | 27/3/7 (universal), 10/0/0 (ε=16), 3/11/9 (random); 16 unsure, 44 decided | human labels | 60 | NONE | Labels missing |
| 32 | §10 | unchanged ≥ 0.755, changed ≤ 0.652 | labels + similarities | 44 | NONE | Similarities computable, labels missing |
| 33 | §10 table | thresholds 0.3/0.5/0.6/0.7: 70.5/86.4/97.7/100%; κ .45/.72/.95/1.00; misses 13/6/1/0 | labels + similarities | 44 | NONE | |
| 34 | §10, §12 | short answers: content-word 28% (text-sim 28%; v 83.0% / 86.9%) | nb03-E answers | 100 (99 for content-word) | CAPS | Content-word metric needs no MiniLM |
| 35 | §11.1 | JPEG q75: 3.0 / 8.5 / 23.0 | nb05 JSON | 200 | AGG | No captions |
| 36 | §11.1 | blur σ1: 6.0 / 5.0 / 29.5 | nb05 | 200 | AGG | No captions |
| 37 | §11.1 | random resize-pad: 6.0 / 4.0 / 21.0 | nb05; also nb07 part A captions (same defense, same seed) | 200 | AGG (CAPS proxy) | nb07 printed the same 6.0 / 4.0 / 21.0, so its captions are a likely match. To confirm |
| 38 | §11.1 | global tile rrp: 1.0 / 2.0 / 10.0 | nb05 | 100 | AGG | The first 100 of the nb06 captions are a candidate proxy |
| 39 | §11.1 | global tile blanked: 23.0 / 20.0 / 80.0 | nb05 | 100 | AGG | |
| 40 | §11.1 | fixed 8/255 column (Fig. 9) | nb05 | 200 / 100 | AGG | |
| 41 | §11.2 | clean cost 2.5% (1.1–5.7) | nb06 JSON k=5 + captions | 200 | CAPS | |
| 42 | §11.2 | fixed v, no defense, 60.5% (53.6–67.0) | nb06 | 200 | CAPS | |
| 43 | §11.2 | fixed v, defended, 1.5% (0.5–4.3) | nb06 | 200 | CAPS | |
| 44 | §11.2 | weak adaptive, defended, 13.0% (9.0–18.4) | nb06 | 200 | CAPS | |
| 45 | §11.2 | strong, 6 epochs, 30.5% (24.5–37.2) | nb06 printed output only | 200 | AGG | Captions overwritten |
| 46 | §11.2 | strong, 12 epochs, 35.5% (29.2–42.3) | nb06 JSON k=71 + captions | 200 | CAPS | Checkpoint says epoch=12 |
| 47 | §11.2 | strong, 12 epochs, no defense, 29.5% | nb06 JSON k=59 | 200 | CAPS | |
| 48 | §11.2 | "50 captions the weak did not, reverse 5" | nb06 captions | 200 | CAPS | Notebook printout (6-epoch run) said 38 / 3 |
| 49 | §11.2 | local cos 0.358 → 0.251; global 0.676 → 0.634 | training log | build set | NONE | Not in any output. The saved run printed 0.272 → 0.261 / 0.645 → 0.641 |
| 50 | §11.2 | emulation 0.9975, same tile order | nb06 cell 8 | 3 images | AGG | |
| 51 | §11.4 | rrp loss 0.543 → 0.541; rate 28.0% → 29.0% | nb07 output (loss); epoch-7 rate only in Fig. 12 ("28%") | | AGG | Figure is stale |
| 52 | §11.4 | rrp strongest 29.0% (23.2–35.6) | nb07 captions `strong adaptive v, rrp defense` vs `clean, rrp defense` | 200 | CAPS | |
| 53 | §11.4 | rrp clean cost 6.0% / 25.0% | nb07 `clean, original encoder` vs `clean, rrp defense` | 200 | CAPS | |
| 54 | §11.4 | global-tile clean cost 2.5% / "--" | nb06 captions | 200 | CAPS | The 0.7 value is computable (Step 6) |
| 55 | §11.4 | global-tile strongest 35.5% | = row 46 | 200 | CAPS | |
| 56 | §11.3, §11.4 | FT-fixed clean cost 3.0% / 8.0% | nb07 | 200 | CAPS | |
| 57 | §11.3 | FT-fixed fixed v 4.0% | nb07 | 200 | CAPS | |
| 58 | §11.3, §11.4 | FT-fixed adaptive 41.0% (34.4–47.9) | nb07 | 200 | CAPS | |
| 59 | §11.3, §11.4 | FT-UAT clean cost 4.5% / 13.5% | nb07 | 200 | CAPS | |
| 60 | §11.3 | FT-UAT fixed v 2.5% | nb07 | 200 | CAPS | |
| 61 | §11.3, §11.4 | FT-UAT adaptive 37.5% (31.1–44.4) | nb07 | 200 | CAPS | |
| 62 | §11.3 | drift 0.971 / 0.953 | nb07 printed | 50 | AGG | |
| 63 | §11.3 | attacker cos 0.675 vs 0.361 | nb07 training log | **build set**, 500 | AGG | Not held-out; no per-image values |
| 64 | §11.4 | the 0.7 column for rrp, FT-fixed, FT-UAT | nb07 captions | 200 | CAPS | |
| 65 | §12 | random passes cos < 0.9 for 91.6% | week1 `emb_fool@0.9` | 500, after preprocessing | AGG | |

Notes on the "known suspects":
- **3.5% vs 4.0%:** two different noise draws (the CUDA RNG in nb02 vs the CPU RNG in nb04 and nb05). Only the 4.0% draw has saved captions.
- **5.0% vs 2%:** nb01 used the 200 images through `caption()` and nb02 the first 100 through `gen_tensor()`. Both share the CUDA noise. The 2% is not traceable, but week1's first 100 images give a proxy.
- **70.0 / 59.0 / 60.5%:** nested prefixes (50 ⊂ 100 ⊂ 200) of the same images. Only the 200-image captions are saved, so the 50 and 100 subsets can be recomputed from them.
- **Global-tile 100- vs 200-image results:** also nested prefixes. The 200-image captions (nb06) can be cut to the first 100 to test whether nb05's 1/2/10% reappears.
- **29.0% vs 28%:** the figure is stale (epoch 7), while the table and JSON are epoch 8.
- **50 / 5:** recomputable from the nb06 captions. The printout in notebook 06 reports 38 / 3 for the 6-epoch v.
