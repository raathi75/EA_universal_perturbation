# Mini-VLM: SigLIP, CLIP, I-JEPA, MAE (same LM, connector and data)

ξ = 10/255, perturbations built on ImageNetV2, images rounded to 8 bits, 95% Wilson intervals.

## COCO val2017, against the human captions

(siglip, clip, ijepa, mae: mini-VLMs at 224 px; smolvlm: SmolVLM-256M with the perturbation of the report; smolvlm-256m, llava-ov-0.5b, qwen2-vl-2b (448 px), qwen2-vl-2b-224 (same model at 224 px): real VLMs, own perturbation run to convergence)

| | siglip | clip | ijepa | mae | smolvlm | smolvlm-256m | llava-ov-0.5b | qwen2-vl-2b | qwen2-vl-2b-224 |
|---|---|---|---|---|---|---|---|---|---|
| CIDEr, clean (quality: must be close) | 1.175 | 1.020 | 0.909 | 0.919 | 0.280 | 0.216 | 0.522 | 0.322 | 0.433 |
| CIDEr, own universal v | 0.693 | 0.348 | 0.265 | 0.121 | 0.071 | 0.025 | 0.127 | 0.537 | 0.456 |
| wrong (far from every human caption), clean | 2.9% (2–4) | 7.1% (6–9) | 15.9% (14–18) | 13.6% (12–16) | 27.4% (25–30) | 23.9% (21–27) | 5.8% (5–7) | 1.6% (1–3) | 4.7% (4–6) |
| wrong (far from every human caption), own universal v | 21.6% (19–24) | 64.1% (61–67) | 75.0% (72–78) | 93.1% (91–95) | 86.6% (84–89) | 93.7% (92–95) | 58.9% (56–62) | 3.7% (3–5) | 11.6% (10–14) |
| wrong (far from every human caption), random ±ξ | 4.2% (3–6) | 6.0% (5–8) | 19.0% (17–22) | 14.1% (12–16) | 30.3% (28–33) | 26.3% (24–29) | 5.4% (4–7) | 2.6% (2–4) | 5.2% (4–7) |
| changed vs clean caption, own universal v | 27.8% (25–31) | 66.9% (64–70) | 75.1% (72–78) | 93.6% (92–95) | 71.7% (69–74) | 88.2% (86–90) | 60.9% (58–64) | 2.2% (1–3) | 10.7% (9–13) |
| changed vs clean caption, random ±ξ | 4.6% (3–6) | 4.6% (3–6) | 9.5% (8–11) | 6.5% (5–8) | 6.0% (5–8) | 2.2% (1–3) | 2.1% (1–3) | 1.0% (1–2) | 2.0% (1–3) |
| CIDEr, v built on the connector outputs | 0.767 | 0.764 | 0.306 | 0.292 | — | — | — | — | — |
| wrong, v built on the connector outputs | 20.4% (18–23) | 24.5% (22–27) | 72.0% (69–75) | 69.8% (67–73) | — | — | — | — | — |
| changed vs clean caption, v built on the connector outputs | 25.7% (23–28) | 26.0% (23–29) | 70.1% (67–73) | 70.6% (68–73) | — | — | — | — | — |
| CIDEr, end-to-end v (likelihood of the clean captions) | 0.711 | 0.726 | — | — | — | — | — | 0.331 | 0.327 |
| wrong, end-to-end v | 26.1% (23–29) | 24.6% (22–27) | — | — | — | — | — | 6.2% (5–8) | 15.9% (14–18) |
| changed vs clean caption, end-to-end v | 30.7% (28–34) | 25.7% (23–28) | — | — | — | — | — | 3.7% (3–5) | 16.5% (14–19) |

## ImageNetV2 held-out images of the SmolVLM report: captions changed (500 images)

| | siglip | clip | ijepa | mae | smolvlm | smolvlm-256m | llava-ov-0.5b | qwen2-vl-2b | qwen2-vl-2b-224 |
|---|---|---|---|---|---|---|---|---|---|
| own universal v | 47.4% (43–52) | 79.2% (75–83) | 80.4% (77–84) | 92.8% (90–95) | 80.0% (76–83) | 91.2% (88–93) | 70.2% (66–74) | 5.6% (4–8) | 15.2% (12–19) |
| random ±ξ | 10.8% (8–14) | 13.8% (11–17) | 17.4% (14–21) | 14.0% (11–17) | 8.4% (6–11) | 5.4% (4–8) | 2.2% (1–4) | 1.0% (0–2) | 3.4% (2–5) |
| v built on the connector outputs | 44.4% (40–49) | 50.8% (46–55) | 77.4% (74–81) | 74.8% (71–78) | — | — | — | — | — |
| end-to-end v (likelihood of the clean captions) | 48.4% (44–53) | 44.6% (40–49) | — | — | — | — | — | 7.8% (6–10) | 23.8% (20–28) |

## Transfer: captions changed on COCO (rows: perturbation built on, columns: model attacked)

| v built on ↓ / model → | siglip | clip | ijepa | mae |
|---|---|---|---|---|
| siglip | 27.8% (25–31) | 14.0% (12–16) | 14.1% (12–16) | — |
| clip | 16.7% (15–19) | 66.9% (64–70) | 12.2% (10–14) | — |
| ijepa | 10.6% (9–13) | 10.8% (9–13) | 75.1% (72–78) | 27.3% (25–30) |
| mae | 7.1% (6–9) | 6.5% (5–8) | 11.7% (10–14) | 93.6% (92–95) |
| SmolVLM-256M | 2.8% (2–4) | 4.2% (3–6) | 5.3% (4–7) | 4.4% (3–6) |

## Token damage on the ImageNetV2 held-out images (mean token cosine, 1 = unchanged)

| v built on ↓ / encoder → | siglip | clip | ijepa | mae |
|---|---|---|---|---|
| siglip | 0.253 | 0.538 | 0.740 | — |
| clip | 0.447 | 0.303 | 0.744 | — |
| ijepa | 0.543 | 0.600 | 0.299 | 0.905 |
| mae | 0.661 | 0.710 | 0.781 | 0.092 |
| SmolVLM-256M | 0.809 | 0.817 | 0.877 | 0.986 |
| random | 0.722 | 0.772 | 0.811 | 0.973 |

## Real VLMs: captions changed on COCO by each perturbation (rows: built on, columns: model attacked)

| v built on ↓ / model → | smolvlm-256m | llava-ov-0.5b | qwen2-vl-2b | qwen2-vl-2b-224 |
|---|---|---|---|---|
| own | 88.2% (86–90) | 60.9% (58–64) | 2.2% (1–3) | 10.7% (9–13) |
| siglip | 4.0% (3–5) | 3.8% (3–5) | 3.5% (3–5) | 15.8% (14–18) |
| clip | 4.1% (3–6) | 3.4% (2–5) | 5.1% (4–7) | 20.5% (18–23) |
| ijepa | 4.4% (3–6) | 4.0% (3–5) | 2.2% (1–3) | 13.4% (11–16) |
| mae | 2.6% (2–4) | 2.9% (2–4) | 1.5% (1–2) | 3.9% (3–5) |
| smolvlm (report) | 90.5% (89–92) | 1.5% (1–2) | 1.0% (1–2) | 1.8% (1–3) |
| smolvlm-256m | (own) | — | 1.2% (1–2) | 1.5% (1–2) |
| llava-ov-0.5b | 2.5% (2–4) | (own) | 3.8% (3–5) | 2.4% (2–4) |
| qwen2-vl-2b | — | — | (own) | 0.6% (0–1) |
| qwen2-vl-2b-224 | — | — | — | (own) |

## Defense: fine-tuning the encoder (as in the report, notebook 07), captions changed

clean cost: clean captions vs those of the original mini-VLM; fixed v / adaptive v: perturbed vs clean captions of the same model. no-defense = the original encoder attacked by the same adaptive procedure (equal attack effort). COCO val (1,000 images) / ImageNetV2 held-out (500 images).

| | siglip | clip | ijepa | mae |
|---|---|---|---|---|
| no-defense, fixed v | 27.8% / 47.4% | 66.9% / 79.2% | 75.1% / 80.4% | 93.6% / 92.8% |
| no-defense, adaptive v | 28.7% / 48.0% | 69.4% / 78.6% | 75.5% / 80.8% | 93.6% / 93.2% |
| FT-fixed, clean cost | 2.0% / 5.8% | 4.2% / 11.6% | 2.9% / 8.2% | 2.0% / 3.4% |
| FT-fixed, fixed v | 6.9% / 16.4% | 7.9% / 17.6% | 7.8% / 12.8% | 7.1% / 15.2% |
| FT-fixed, adaptive v | 64.6% / 70.4% | 53.0% / 64.4% | 57.3% / 63.6% | 96.4% / 94.8% |
| FT-UAT, clean cost | 1.3% / 4.6% | 4.4% / 11.4% | 3.9% / 7.8% | 1.7% / 4.2% |
| FT-UAT, fixed v | 8.0% / 14.8% | 9.4% / 21.2% | 15.3% / 24.8% | 5.9% / 15.4% |
| FT-UAT, adaptive v | 67.8% / 72.6% | 51.6% / 63.2% | 76.2% / 77.0% | 85.1% / 85.0% |

## Post-training with the encoder frozen or unfrozen (stage 2), clean quality and robustness

Cells: captions changed on COCO val / on the ImageNetV2 held-out images. fixed v = perturbation of the original encoder (for the frozen model it is also its own, recomputed one); adaptive v = recomputed against the model.

| | siglip frozen | siglip unfrozen |
|---|---|---|
| CIDEr, clean | 1.175 | 1.200 |
| wrong, clean (COCO) | 2.9% (2–4) | 3.3% (2–5) |
| fixed v (original encoder) | 27.8% / 47.4% | 24.1% / 43.6% |
| adaptive v (recomputed, tokens) | 27.8% / 47.4% | 22.5% / 42.8% |
| adaptive v (recomputed, connector outputs) | 25.7% / 44.4% | 19.9% / 37.8% |
| random ±ξ | 4.6% / 10.8% | 4.5% / 8.2% |
| CIDEr, adaptive v (tokens) | 0.693 | 0.758 |

## Convergence of the attacks (epoch kept / epochs run, score on the build images: feature cosine, or caption likelihood for the end-to-end v)

| | siglip | clip | ijepa | mae | smolvlm | smolvlm-256m | llava-ov-0.5b | qwen2-vl-2b | qwen2-vl-2b-224 |
|---|---|---|---|---|---|---|---|---|---|
| token v | 37 / 37, 0.224 | 35 / 36, 0.277 | 25 / 25, 0.278 | 35 / 36, 0.079 | — | 42 / 44, -0.193 | 48 / 48, 0.113 | 25 / 25, 0.873 | 24 / 25, 0.838 |
| connector-output v | 35 / 37, 0.485 | 44 / 45, 0.671 | 31 / 31, 0.650 | 52 / 54, 0.381 | — | — | — | — | — |
| end-to-end v | 28 / 30, 0.180 | 28 / 30, 0.204 | — | — | — | — | — | 21 / 23, 0.360 | 39 / 41, 0.228 |
| adaptive v, no-defense | 13 / 13, 0.215 | 9 / 11, 0.273 | 9 / 9, 0.276 | 9 / 9, 0.074 | — | — | — | — | — |
| adaptive v, FT-fixed | 29 / 29, 0.449 | 27 / 27, 0.540 | 42 / 42, 0.495 | 23 / 23, 0.551 | — | — | — | — | — |
| adaptive v, FT-UAT | 28 / 28, 0.460 | 49 / 51, 0.535 | 19 / 19, 0.463 | 28 / 28, 0.403 | — | — | — | — | — |

## ImageNet accuracy read from the captions (500 ImageNetV2 held-out images, top-1 / top-5, %)

Each caption is matched (MiniLM) with the 1000 class names; captions are generic, so compare columns and rows, not with a classifier.

| | siglip | clip | ijepa | mae | smolvlm | smolvlm-256m | llava-ov-0.5b | qwen2-vl-2b | qwen2-vl-2b-224 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 14.4 / 30.6 | 7.6 / 17.2 | 5.2 / 15.4 | 5.0 / 11.8 | 11.2 / 24.2 | 14.8 / 27.8 | 36.4 / 52.0 | 42.8 / 58.0 | 41.2 / 58.0 |
| own universal v | 9.2 / 18.6 | 3.4 / 7.6 | 1.8 / 5.0 | 0.4 / 1.0 | 1.8 / 2.8 | 1.0 / 1.2 | 11.0 / 17.2 | 38.6 / 54.2 | 31.2 / 45.2 |
| end-to-end v | 7.8 / 16.2 | 4.8 / 11.2 | — | — | — | — | — | 42.8 / 54.8 | 31.8 / 46.8 |
| no-defense, adaptive v | 9.8 / 18.8 | 3.6 / 6.4 | 2.2 / 4.8 | 0.6 / 1.2 | — | — | — | — | — |
| random ±ξ | 15.0 / 28.8 | 8.0 / 16.0 | 4.4 / 12.6 | 5.6 / 10.6 | 12.4 / 23.6 | 14.4 / 26.2 | 35.2 / 50.2 | 43.6 / 59.6 | 37.6 / 54.0 |
| FT-fixed, clean | 15.0 / 30.2 | 9.0 / 17.0 | 5.2 / 15.0 | 5.4 / 12.2 | — | — | — | — | — |
| FT-fixed, fixed v | 15.0 / 27.6 | 6.8 / 15.8 | 5.0 / 11.8 | 5.2 / 11.0 | — | — | — | — | — |
| FT-fixed, adaptive v | 6.2 / 11.2 | 3.6 / 8.4 | 3.2 / 7.0 | 0.2 / 0.6 | — | — | — | — | — |
| FT-UAT, clean | 15.8 / 28.6 | 7.2 / 17.0 | 5.0 / 14.4 | 5.8 / 12.4 | — | — | — | — | — |
| FT-UAT, fixed v | 13.0 / 26.6 | 7.0 / 15.8 | 4.8 / 11.0 | 5.0 / 11.2 | — | — | — | — | — |
| FT-UAT, adaptive v | 6.2 / 11.4 | 3.6 / 8.0 | 1.6 / 4.6 | 0.8 / 2.4 | — | — | — | — | — |
