# Edits to the draft from the statistical audit

Draft: `Universal_Adversarial_Perturbations_on_a_Tiny_Vision_Language_Model.pdf` (Nov 12, 2026).

This list is meant to be applied to the draft's source, which isn't in the repository (only the compiled PDF is). Each edit gives:
- where it goes;
- the current text, quoted from the PDF;
- the replacement;
- the reason, with a pointer to `RECAP.md`, `all_numbers.csv` or `paired_tests.csv`.

The edits are ordered by section. The ones marked **[conclusion]** change what the report claims. The others are precision fixes.

---

## Abstract

**A1 [conclusion]**

- **Current:** "every defense helps, none clearly dominates, and the strongest attacker still changes between 29% and 41% of captions."
- **Replace with:** "every defense helps, none is best on every count (random resizing leaves the least residual attack but disturbs the most clean captions), and the strongest attacker still changes between 29% and 41% of captions."
- **Why:** Paired tests show random resize-and-pad is significantly more robust than both fine-tuned encoders (McNemar p = 0.027 and 0.003). At the 0.7 threshold, it is also significantly more costly on clean captions (p < 0.001). RECAP §2.

## §4.4 Evaluation protocol

**A2**

- **Current:** "every rate comes with a 95% Wilson confidence interval in the later experiments,"
- **Replace with:** "every rate comes with a 95% Wilson confidence interval,"
- **Why:** Every rate in the report now has one (see the tables below and `results/latex_rows.tex`).

**A3 (add after the sentence about the 0.5 threshold)**

> "An empty answer has a similarity of about 0.1 with any caption and is therefore counted as changed; this happens for one to three images per perturbed run."

- **Why:** RECAP §1, item 9.

## §4.5 How to read the numbers

**A4** Replace the paragraph that starts "A 95% confidence interval (CI) tells us…" up to "…should not be ranked against each other." with:

> "Every rate is reported with a 95% Wilson interval; when two conditions are evaluated on the same images, we compare them with an exact McNemar test on the images whose outcome differs between the two conditions, and we measure agreement between the metric and the human annotator with Cohen's κ. These intervals and tests reflect only which held-out images were evaluated: each perturbation and each fine-tuned encoder was trained once, with a single seed, so they do not capture variation from training randomness. The Wilson interval needs only the count of changed captions and the number of images: 121 changed captions out of 200, for example, give 60.5% with an interval of 53.6-67.0%. It shrinks as the number of images grows, from roughly ±13 points with 50 images to ±9 with 100 and ±7 with 200, which is why small runs of the same experiment can give visibly different numbers. Overlapping intervals do not by themselves mean that two conditions are equivalent: when both were run on the same images, the paired McNemar test is the one we use to rank them. Token cosines and PSNR are reported as mean, standard deviation and number of images."

- **Why:** This is the methods sentence the audit asked for. It also corrects the old rule, "intervals that overlap should not be ranked", which §11.4 relied on and which the paired test contradicts.

## §5 Table (after preprocessing) and §6 settings table

**A5** Add a token-cosine footnote to the table:

> "Token cosine: mean (SD) over 500 held-out images after preprocessing, −0.051 (0.256) for v and 0.846 (0.041) for random noise; through the real pipeline, mean (SD) over the first 50 images, 0.142 (0.342) with splitting off and 0.688 (0.051) averaged over 17 tiles with splitting on."

- **Why:** The 0.142 and 0.688 come from 50 images, not the 200 the table header implies. RECAP §1 item 2 and §4.

**A6** In the "Random ±ξ" column of the §6 table, add a footnote to 3.5% and 2.0%:

> "Random noise drawn on the GPU, as in Section 5; the budget sweep of Section 9 uses a draw made on the CPU with the same seed, which gives 4.0% in the same setting."

- **Why:** Two noise draws, both correct and both now reproduced from per-image captions. RECAP §1, item 3.

**A7** In the "splitting reduces the attack by about ten points" sentence, after "…about ten points", insert:

> "(71.5% vs 60.5% on the same 200 images; McNemar p < 0.001)"

- **Why:** 28 vs 6 discordant images, p = 0.0002. `paired_tests.csv`.

## §6 Partial attractor

**A8**

- **Current:** "the phrase "white shirt" appeared in roughly one perturbed caption in six,"
- **Replace with:** "the phrase "white shirt" appeared in 27 of the 200 perturbed captions (13.5%, roughly one in seven), always together with a person,"
- **Why:** The count is 27/200 = 13.5% [9.4, 18.9]. All 27 also mention a person. RECAP §1, item 5.

**A9** After "doubled, from 14% to 28%,", add:

> "(counting captions that contain "person", "man" or "woman")"

- **Why:** §9 uses a different definition. RECAP §1, item 7.

## §7 Tile table

**A10** Change the table header to "Splitting on, 50 images, mean token cosine (SD)", with these values:
- Universal v: local tiles 0.723 (0.038), global tile 0.142 (0.342)
- Random ±ξ: local tiles 0.771 (0.021), global tile 0.865 (0.038)

**A11** After "The global tile is damaged exactly as much as the single tile in the splitting-off setting (0.142 in both cases)", add:

> "; image by image the two values agree to within 3×10⁻⁶, as they should, since with splitting off the single tile is the global view"

## §8 Transfer table

**A12**

- In the column header, change "Token cosine (v)" to "Token cosine (v), mean (SD)".
- Set the values to 0.142 (0.342), 0.706 (0.101) and 0.871 (0.039).
- Add the footnote: "Token cosine on the first 50 images for 256M and the first 30 for 500M and 2.2B."
- Add footnote A6 to the 256M random-noise 3.5%.
- **Why:** RECAP §1 item 2 and §4.

**A13** Keep "essentially unaffected (7% against 3%, within the noise of 100 images)". It is supported: 7.0% [3.4, 13.7] against 3.0% [1.0, 8.5].

## §9 Budget sweep table

**A14**

- Change the 16/255 "white shirt" value from **18%** to **18.5%**.
- Rename the column to "Captions with "person" or "white shirt"" (already the case), and add the footnote: "a broader definition than the person count of Section 6, so the 10/255 value is not directly comparable with the 28% there".
- Add the footnote: "Token cosine and PSNR: mean over the first 50 images; SD 0.15–0.36 for the cosine and 0.26–0.33 dB for PSNR."
- **Why:** 37/200 = 18.5%, which Python's round-half-to-even printed as 18. RECAP §1, items 6 and 7.

**A15 (optional, one sentence after the table)**

> "Across images the attack is far less uniform than noise: the standard deviation of the token cosine is 0.28–0.36 for v from 8/255 upwards, against 0.02–0.04 for random noise, so the mean hides images that are almost untouched alongside images whose features are destroyed."

## §11.1 Test-time defense table

**A16** Add 95% Wilson intervals to every cell. The values are in `results/latex_rows.tex`, block "Section 11.1". All the draft's numbers are confirmed.

## §11.2 Strong adaptive attack

**A17** The 6-epoch row, "Strong adaptive v, defended, after 6 epochs 30.5% 24.5-37.2%", has a footnote:

> "From an earlier run whose captions and perturbation were not kept; the value is the printed result of that run."

- **Why:** NOT TRACEABLE. It was overwritten by the 12-epoch rerun. RECAP §1, item 1.

**A18** In the paragraph that starts "The weak attacker had underestimated the threat…", make two changes.

- **Current:** "the 12-epoch attack breaks 50 captions that the weak one did not, while the reverse happens five times."
  **Replace with:** "the 12-epoch attack breaks 50 captions that the weak one did not, while the reverse happens five times (exact McNemar p < 10⁻⁹)."
- **Current:** "Damage to the local tiles kept growing (token cosine falling from 0.358 to 0.251) while the defended global tile barely moved (0.676 to 0.634), so the defense does protect…"
  **Replace with:** "During training, damage to the local tiles kept growing while the defended global tile barely moved, so the defense does protect…"
  - **Why:** These four values appear in no saved file. The log saved in notebook 06 is from a different run. If you keep numbers here, label them as coming from an unsaved run.

**A19**

- **Current:** "Doubling the training from 6 to 12 epochs added five points while the loss fell only slightly, so the attack is close to, though not provably at, its ceiling."
- **Replace with:** "Doubling the training from 6 to 12 epochs added five points (30.5% to 35.5%; the two runs cannot be compared image by image because the 6-epoch captions were not kept), so the attack is probably close to, though not provably at, its ceiling."

**A20 [conclusion]**

- **Current:** "Interestingly, the 12-epoch perturbation does better against the defended model (35.5%) than against the undefended one (29.5%), because it has specialised to the defended pipeline; an attacker has to commit to one of the two."
- **Replace with:** "The 12-epoch perturbation is about as effective against the undefended model (29.5%) as against the defended one (35.5%); on the same 200 images the difference is not significant (exact McNemar p = 0.08)."
- **Why:** 26 vs 14 discordant images, p = 0.081. `paired_tests.csv`.

**A21** Add the 12-epoch undefended row to the §11.2 table, since it is quoted in the text: "Strong adaptive v (12 epochs), no defense | 29.5% | 23.6-36.2%".

## §11.3 Fine-tuning

**A22**

- **Current:** "since after the same seven epochs the attacker only reaches a token cosine of 0.675 against it, versus 0.361 against FT-fixed,"
- **Replace with:** "since after the same seven epochs the attacker only reaches a token cosine of 0.675 against it, versus 0.361 against FT-fixed (final training loss on the 500 images used to build the perturbation),"
- **Why:** These values are the attacker's training loss on the build images, not measured on held-out images. RECAP §1, item 2.

**A23** After "(clean-token cosine 0.971 for FT-fixed and 0.953 for FT-UAT", add "on 50 held-out images".

## §11.4 Ranking

**A24** In the ranking table:
- in the global-tile row, replace "2.5% / --" with "2.5% / 15.5%";
- add 95% intervals to the clean-cost cells (`results/latex_rows.tex`, block "Section 11.4").
- **Why:** The 0.7 value can be computed from the saved captions: 31/200, interval [11.1, 21.2].

**A25 [conclusion]** Replace the paragraph that starts "Three conclusions follow…" with:

> "Three conclusions follow, and the first one replaces our earlier headline. No defense is best on every count. Random resize-and-pad leaves the least residual attack (29.0%), significantly less than FT-UAT (37.5%) and FT-fixed (41.0%) on the same 200 images (exact McNemar p = 0.027 and 0.003), but it is also the one that disturbs normal use the most: it changes only 6% of clean captions at our main threshold but 25% at 0.7, significantly more than FT-UAT (13.5%) and FT-fixed (8.0%; p < 0.001 for both), because it alters many details of normal answers. The two fine-tuned encoders do not differ in residual attack (p = 0.39). The global-tile defense, which runs with splitting on and therefore cannot be compared on equal terms, sits in the same range: its residual attack (35.5%) does not differ significantly from any of the others, and it costs as little as the fine-tuned encoders on clean captions at the main threshold (2.5%) without any training, though at 0.7 it alters more clean captions than FT-fixed (15.5% vs 8.0%, p = 0.03). Our tiling-based defense is therefore a useful, measured contribution rather than the winner we first thought it was. Finally, every defense is partial, since the strongest attacker still changes between 29% and 41% of captions in every case, which is what the literature on adaptive attacks would predict. Each defense was attacked by a different adaptive attacker, so these comparisons are between the strongest attacks we tested, not between proven worst cases."

- **Why:** `paired_tests.csv`, RECAP §2. "Cheapest on clean captions" is not supported: at 0.5 the global tile does not differ from any defense (all p ≥ 0.12), and at 0.7 it is worse than FT-fixed.

**A26** Figure 12: redraw it. Random resize-and-pad shows 28%, the stale epoch-7 value. The table and the saved captions give 29.0%. RECAP §1, item 4.

## §12 What we got wrong (table)

**A27**
- In the "dominant caption" row, replace "A partial attractor for roughly one image in six" with "A partial attractor for roughly one image in seven".
- In the "global-tile defense is the best" row, replace "No single best defense; the global-tile one is the cheapest with comparable robustness" with "No defense is best on every count; the global-tile one is comparably robust and nearly free on clean captions at the main threshold".

## §14 Limitations

**A28** After "Every perturbation was trained once, with a single seed,", insert:

> "so our intervals and paired tests reflect only the choice of evaluation images, not training randomness;"

## §15 Conclusion

**A29**

- **Current:** "Our own tiling-based defense remains attractive because it is nearly free on clean inputs, but it is one reasonable option among several rather than a solution."
- **Replace with:** "Our own tiling-based defense remains attractive because it needs no training and costs little on clean inputs, but it is one reasonable option among several rather than a solution; the most robust option we tested, random resizing, is also the one that disturbs normal answers the most."

## §10 Human check (pending)

These edits wait for the labels file:
- Report κ as "κ = 0.72 (one annotator, 44 decided pairs)", without an interval.
- Give agreement with its Wilson interval: 38/44 = 86.4% [73.3, 93.6].
- The other counts with intervals are in RECAP §3.
