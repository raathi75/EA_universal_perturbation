"""Builds the result notebooks 08-12 in smol_exp/ (next to notebooks 01-07), executes them and stores their outputs.

  python make_notebooks.py               build and execute all
  python make_notebooks.py 08 10         only these
  python make_notebooks.py --no-exec     write the notebooks without outputs

The notebooks read the result files of the experiment scripts; see nb_utils.py. Executing them here (instead of with
Jupyter) only needs the python of the experiment environment: each code cell is run in one namespace per notebook,
its printed text, Markdown tables (nb_utils.show_table) and matplotlib figures become the cell outputs.
Notebook 12 runs SmolVLM-256M on the CPU (a few minutes).
"""
import base64
import contextlib
import io
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
NB_DIR = HERE.parent          # smol_exp/


def md(s):
    return ('markdown', s.strip('\n'))


def code(s):
    return ('code', s.strip('\n'))


SETUP = code(r'''
import os, sys
sys.path.insert(0, os.path.abspath('minivlm'))   # run from smol_exp/, where notebooks 01-07 are
import numpy as np
import matplotlib.pyplot as plt
from nb_utils import *
use_style()
print('result files read from', RESULTS_DIR)
''')

# =====================================================================================================================
NB08 = ('08_minivlm_encoder_comparison.ipynb', [
md(r'''
# Mini-VLMs: does the vision encoder decide universal vulnerability?

Notebooks 01–07 studied one model, SmolVLM-256M. Here, four small VLMs differ **only in their frozen vision encoder**:

| encoder | pre-training objective | size |
|---|---|---|
| SigLIP-SO400M/14 | contrastive image–text (sigmoid) | ~400M |
| CLIP ViT-L/14 | contrastive image–text (softmax) | ~300M |
| I-JEPA ViT-H/14 | predicts hidden parts in feature space | ~630M |
| MAE ViT-H/14 | reconstructs hidden pixels | ~630M |

Each mini-VLM = frozen encoder (224 px, 256 tokens) → connector (2×2 pixel shuffle + MLP, trained) → SmolLM2-135M-Instruct (LoRA), trained on COCO captions, one epoch per stage. I-JEPA and MAE share architecture and data: they are the cleanest pair.

**Attack.** The label-free universal perturbation of notebook 01 (maximise the damage to the encoder tokens, ℓ∞, ξ = 10/255, built on the same 500 ImageNetV2 images), but **run to convergence** (Adam, step halved on plateaus, best v kept) and repeated with **3 seeds**. **Evaluation:** the 500 held-out ImageNetV2 images and 1,000 COCO val images. A caption is *changed* when its MiniLM similarity to the clean caption is below 0.5, as in notebook 01; *wrong* compares it with the 5 human COCO captions.

**This notebook does not rerun the attacks** (hours of GPU each). It reads the files they wrote in `minivlm/results/` and recomputes tables, statistics and figures. Commands that produced them (school GPUs):
- `minivlm/run_minivlm.sh <encoder>`: trains the mini-VLM, builds v, evaluates
- `minivlm/run_converged.sh <encoder>`: attacks to convergence, seeds, transfers
- `minivlm/run_endtoend.sh <encoder>`: end-to-end attack on the caption likelihood

The full write-up, with a confidence level for each claim, is `minivlm/REPORT_EN.md` (Parts I, II, V).
'''),
SETUP,
md(r'''
## 1. Same recipe, four encoders
'''),
code(r'''
rows = []
for e in ENC:
    inv = [inv2(e, own(e, s))['rate'] for s in ['', 'seed 1', 'seed 2']]
    coc = [coco(e, own(e, s))['rate'] for s in ['', 'seed 1', 'seed 2']]
    noise = [inv2(e, 'random ±xi' + d)['rate'] for d in ['', ', draw 1', ', draw 2', ', draw 3', ', draw 4']]
    rows.append([NAMES[e], pct(coco(e, 'clean', 'wrong'), False),
                 ' / '.join('%.1f' % r for r in coc) + ' → **%.1f**' % np.mean(coc),
                 ' / '.join('%.1f' % r for r in inv) + ' → **%.1f**' % np.mean(inv),
                 '%.1f – %.1f' % (min(noise), max(noise)),
                 '%.3f' % load(e, 'uap_xi10_tokens.json')['token_cos'][own(e)]])
show_table(['encoder', 'clean, wrong (COCO)', 'changed, COCO, seeds 0/1/2 → mean (%)',
            'changed, ImageNetV2, seeds 0/1/2 → mean (%)', 'random ±ξ, 5 draws (%)', 'token cosine (seed 0)'], rows)
'''),
code(r'''
fig, ax = plt.subplots(figsize=(7, 3.1))
x = np.arange(len(ENC))
bars(ax, x - 0.27, [coco(e, own(e)) for e in ENC], BLUE, 'own v, COCO val', 0.26)
bars(ax, x, [inv2(e, own(e)) for e in ENC], ORANGE, 'own v, ImageNetV2 held-out', 0.26)
bars(ax, x + 0.27, [inv2(e, 'random ±xi') for e in ENC], GRAY, 'noise ±ξ, ImageNetV2', 0.26)
ax.set_xticks(x, [NAMES[e] for e in ENC]); ax.set_ylim(0, 100)
ax.set_ylabel('captions changed (%)'); ax.legend(loc='upper left', fontsize=8)
ax.set_title('Mini-VLMs, same recipe, ξ = 10/255, seed 0 (95 % Wilson CI)', fontsize=9, loc='left')
plt.show()
'''),
md(r'''
**Reading.** With everything else fixed, SigLIP resists far better than the three others, and the ranking SigLIP ≪ CLIP ≈ I-JEPA < MAE holds for every seed (report I.1, confidence A). Random noise of the same size changes only a few percent of captions, so the effect comes from the direction of v, not from its norm. **I-JEPA, the predictive encoder that motivated the extension, is not protected.**

The last column shows that damage to the tokens does not predict damage to the captions: SigLIP, CLIP and I-JEPA destroy their tokens about equally, yet change very different shares of captions (report I.3).

## 2. Ten epochs, the budget of notebooks 01–07, gave the wrong ranking
'''),
code(r'''
rows = []
for e in ENC:
    h = load(e, 'uap_xi10_history.json')
    old = 'old_10epochs/' + e
    rows.append([NAMES[e], pct(coco(old, own(e)), False), pct(coco(e, own(e)), False),
                 pct(inv2(old, own(e)), False), pct(inv2(e, own(e)), False), '%d (best: %d)' % (h['epoch'], h['best_epoch'])])
show_table(['encoder', 'COCO: 10 epochs', 'COCO: convergence', 'ImageNetV2: 10 epochs', 'ImageNetV2: convergence',
            'epochs run'], rows, 'captions changed, token v, seed 0')

fig, ax = plt.subplots(figsize=(7, 2.8))
for e, col in zip(ENC, [BLUE, ORANGE, AQUA, YELLOW]):
    h = load(e, 'uap_xi10_history.json')['history']
    ax.plot([r['epoch'] for r in h], [r['build_score'] for r in h], color=col, lw=2, label=NAMES[e])
ax.axvline(10, color=GRAY, ls=':', lw=1); ax.text(10.3, 0.95, 'budget of notebooks 01–07', fontsize=7.5, color=INK2)
ax.set_xlabel('epoch'); ax.set_ylabel('token cosine (build images)'); ax.legend(fontsize=8)
plt.show()
'''),
md(r'''
After 10 epochs, MAE looked *less* vulnerable than I-JEPA. Its attack hovered for several epochs, then collapsed abruptly to the strongest perturbation of the study (the drop in the curve). **A ranking read from short attacks can be reversed**, which is why every attack in this extension runs to convergence (report I.4, confidence A).

## 3. Several attack objectives: none dominates

Besides the token objective, two other universal attacks:
- *connector output*: damage what the language model actually receives;
- *end-to-end*: lower the likelihood of each image's clean caption through the whole model.
'''),
code(r'''
VARIANTS = [('token', ''), ('connector output', 'connector-output'), ('end-to-end caption', 'caption loss')]
rows = []
for e in ENC:
    r = {name: inv2(e, own(e, var)) if has(e, own(e, var)) else None for name, var in VARIANTS}
    best = max((v['rate'], name) for name, v in r.items() if v)
    rows.append([NAMES[e]] + [pct(r[name]) for name, _ in VARIANTS] + ['%.1f %% (%s)' % best])
show_table(['encoder'] + ['%s v' % n for n, _ in VARIANTS] + ['maximum'], rows,
           'ImageNetV2 held-out, captions changed (seed 0)')
'''),
md(r'''
The end-to-end attack, which optimises the caption itself, is **not** an upper bound: for CLIP it is much weaker than the token attack. A robustness estimate must take the maximum over several objectives (report II.1, confidence A).

## 4. Replacement: the same caption for (almost) every image

Notebook 03 found a partial attractor in SmolVLM (about 1 caption in 6 converging to the same text). Count, on COCO, how many images receive the most frequent caption:
'''),
code(r'''
rows, top_v, top_c = [], [], []
for m in ENC + REAL:
    c = captions(m, 'coco')
    nv, cv = dominant(c[own(m)])
    nc, _ = dominant(c['clean'])
    top_v.append(nv); top_c.append(nc)
    rows.append([NAMES[m], nc, '**%d**' % nv, '“%s”' % cv[:90]])
show_table(['model', 'clean: top caption count', 'under own v', 'most frequent caption under v'], rows,
           'COCO val, 1000 images')

fig, ax = plt.subplots(figsize=(7, 3.0))
models = ENC + REAL
y = np.arange(len(models))[::-1]
ax.barh(y + 0.18, top_v, 0.34, color=BLUE, label='under its own universal perturbation')
ax.barh(y - 0.18, top_c, 0.34, color=GRAY, label='clean images')
ax.set_yticks(y, [NAMES[m] for m in models]); ax.set_xlim(0, 1000)
ax.set_xlabel('COCO images (out of 1000) given the most frequent caption')
ax.grid(axis='x', color=GRID); ax.grid(axis='y', visible=False); ax.legend(loc='lower right', fontsize=8)
plt.show()
'''),
md(r'''
MAE gives the same caption to the vast majority of COCO images: its perturbation does not just degrade the description, it **replaces** it. The other models are affected differently; notebook 11 shows that most perturbations instead **inject** a word or a style.

## 5. What the perturbations look like
'''),
code(r'''
fig, axes = plt.subplots(2, 4, figsize=(8, 4.6))
for ax, m in zip(axes.flat, ENC + REAL):
    v = load_v(m)
    ax.imshow(v_image(v), interpolation='nearest')
    ax.set_title('%s (%d px)' % (NAMES[m], v.shape[-1]), fontsize=8); ax.axis('off')
fig.suptitle('Own universal perturbations (token v), contrast stretched: gray = 0, saturated = ±ξ', fontsize=9)
plt.show()
'''),
md(r'''
The perturbations of the two image–text encoders (SigLIP, CLIP) contain letter-like shapes, words and faces; those of I-JEPA and MAE, which never saw text in pre-training, are periodic textures. This echoes the letter-like shapes of SmolVLM's perturbation in notebook 01, but it is a visual observation, not a measurement (report I.5, confidence B).

## 6. Transfers between mini-VLMs
'''),
code(r'''
SRC = ENC + ['SmolVLM-256M, 512 -> 224']
rows = []
for s in SRC:
    row = [NAMES.get(s, 'SmolVLM (notebook 01 v, resized)')]
    for t in ENC:
        k = own(t) if s == t else 'universal v (%s), transfer' % s
        row.append(('**%s**' if s == t else '%s') % pct(inv2(t, k), False) if has(t, k) else '—')
    rows.append(row)
show_table(['v built on ↓ / applied to →'] + [NAMES[t] for t in ENC], rows,
           'ImageNetV2 held-out, captions changed (diagonal: own v)')
'''),
md(r'''
Transfers are partial and asymmetric. The perturbation of notebook 01, resized from 512 to 224 px, is mostly lost: resizing alone destroys much of a universal perturbation (notebook 10 shows this on a single model).

## 7. Unfreezing the encoder (SigLIP only)

The mini-VLM is trained again with the encoder unfrozen, then attacked with the same recipe.
'''),
code(r'''
rows = []
for m in ['siglip', 'siglip-unfrozen']:
    rows.append([NAMES[m], pct(inv2(m, own(m))), pct(coco(m, own(m))), pct(inv2(m, 'random ±xi'))])
show_table(['mini-VLM', 'changed, ImageNetV2', 'changed, COCO', 'random ±ξ, ImageNetV2'], rows)
'''),
md(r'''
Unfreezing the encoder does not make the model more vulnerable; given SigLIP's spread across seeds (section 1), the gap is not significant (report V.1, confidence C; one encoder only).

## Takeaways

| conf. | finding |
|---|---|
| A | With the same recipe, the encoder decides universal vulnerability: SigLIP ≪ CLIP ≈ I-JEPA < MAE. A predictive encoder (I-JEPA) is not protected. |
| A | Short attacks can reverse a ranking (MAE at 10 epochs). Attacks must run to convergence. |
| A | No attack objective dominates: take the maximum over several. |
| A | A universal perturbation can *replace* the caption (MAE). |
| B | Perturbations of image–text encoders contain text- and face-like patterns. |
| C | The I-JEPA/MAE gap as an effect of the pre-training objective; effect of unfreezing. |

**Limits.** Small models (135M language model, one training epoch per stage), a single budget (10/255), one prompt, a single training seed per mini-VLM. The "changed" threshold was validated by hand only on SmolVLM captions (earlier report).
'''),
])

# =====================================================================================================================
NB09 = ('09_defense_on_four_encoders.ipynb', [
md(r'''
# Fine-tuning the encoder as a defense, on four encoders

Notebook 07 defended SmolVLM-256M by fine-tuning its vision encoder. It concluded that a defense that blocks the *fixed* perturbation can fail against an attacker who recomputes v against the defended model, and that robustness measured inside the encoder overstates robustness in the answers. Here, the same two defenses on each mini-VLM of notebook 08:
- **FT-fixed**: fine-tune the encoder on images carrying the fixed universal v;
- **FT-UAT**: universal adversarial training, v updated during fine-tuning;
- both: 3 epochs on the 500 build images, with an anchoring term that keeps the clean features close to the original ones; connector and language model unchanged.

Each defended model then faces an **adaptive attacker**: a new universal v computed against it, to convergence.

Files: `results/<encoder>/defense_ft.json`, `defense_ft_captions.json`, `defense_mechanism.json`. Commands: `minivlm/run_defense.sh <encoder>`, then `python defense_mechanism.py --encoder <encoder>`. Report: Part IV.
'''),
SETUP,
md(r'''
## 1. Clean cost, fixed v, adaptive v
'''),
code(r'''
VARS = ['no-defense', 'FT-fixed', 'FT-UAT']
def ft(e, var, cond, ds='imagenetv2'):
    r = load(e, 'defense_ft.json')['variants'][var][ds][cond]
    return r['changed@0.5']['all'] if ds == 'imagenetv2' else r['changed']
rows = []
for e in ENC:
    for var in VARS:
        rows.append([NAMES[e] if var == 'no-defense' else '', var, pct(ft(e, var, 'clean cost'), False),
                     pct(ft(e, var, 'fixed v')), '**%s**' % pct(ft(e, var, 'adaptive v')),
                     pct(ft(e, var, 'adaptive v', 'coco'), False)])
show_table(['encoder', 'defense', 'clean captions changed by the defense', 'fixed v', 'adaptive v',
            'adaptive v, COCO'], rows, 'ImageNetV2 held-out, captions changed')
'''),
code(r'''
fig, axes = plt.subplots(1, 4, figsize=(8, 2.9), sharey=True)
for ax, e in zip(axes, ENC):
    xx = np.arange(3)
    bars(ax, xx - 0.19, [ft(e, v, 'fixed v') for v in VARS], BLUE, 'fixed v', 0.36)
    bars(ax, xx + 0.19, [ft(e, v, 'adaptive v') for v in VARS], ORANGE, 'adaptive v', 0.36)
    ax.set_xticks(xx, ['no\ndefense', 'FT-\nfixed', 'FT-\nUAT'], fontsize=7.5)
    ax.set_title(NAMES[e], fontsize=9); ax.set_ylim(0, 100)
axes[0].set_ylabel('ImageNetV2 captions changed (%)'); axes[0].legend(loc='upper left', fontsize=7.5)
plt.show()
'''),
md(r'''
The pattern of notebook 07 holds on every encoder. The defense changes 3 to 12 % of clean captions and neutralises the fixed v, but **the adaptive v gets through**: 63 to 95 % of captions still change.

## 2. Against the adaptive attacker, is the defended model better than no defense?

Paired test on the same 500 images: each model's caption under its adaptive v is compared with **its own** clean caption, then McNemar compares the defended model with the undefended one. The caption similarities are recomputed here (MiniLM).
'''),
code(r'''
rows = []
for e in ENC:
    c = load(e, 'defense_ft_captions.json')
    flags = {v: changed_flags(c['imagenetv2: %s: clean' % v], c['imagenetv2: %s: adaptive v' % v]) for v in VARS}
    for v in VARS[1:]:
        only_nd, only_def, p = mcnemar(flags['no-defense'], flags[v])
        verdict = 'defense worse' if only_def > only_nd and p < 0.05 else ('defense better' if p < 0.05 else 'no significant difference')
        rows.append([NAMES[e], v, '%.1f %%' % (100 * flags['no-defense'].mean()), '%.1f %%' % (100 * flags[v].mean()),
                     '%d vs %d' % (only_nd, only_def), '%.1g' % p, verdict])
show_table(['encoder', 'defense', 'no defense, adaptive v', 'defended, adaptive v',
            'discordant pairs (only no-def. / only def.)', 'McNemar p', ''], rows)
'''),
md(r'''
For CLIP and I-JEPA (FT-fixed), and MAE (FT-UAT), the defense lowers the adaptive attack by 8 to 17 points, but most captions still change. For SigLIP, the most robust encoder without defense, the defended model is **more** vulnerable to its adaptive perturbation than the undefended one (report IV.3, confidence A). A defense must be judged against the attacker who knows it.

## 3. Where it fails: the encoder is protected, the language model's input is not

For each variant, cosine between clean and perturbed features (500 held-out images), at the encoder output (*tokens*) and at the connector output, which is what the language model reads. Damage = 1 − cosine.
'''),
code(r'''
rows = []
for e in ENC:
    d = load(e, 'defense_mechanism.json')
    for v in VARS:
        r = d['variants'][v]
        rows.append([NAMES[e] if v == 'no-defense' else '', v] +
                    ['%.2f' % (1 - r[a][k]) for a in ['fixed v', 'adaptive v'] for k in ['token_cos', 'connector_cos']])
    if 'note' in d:
        print('%s: %s' % (NAMES[e], d['note']))
show_table(['encoder', 'defense', 'fixed v: token damage', 'fixed v: connector damage', 'adaptive v: token damage',
            'adaptive v: connector damage'], rows)

fig, axes = plt.subplots(1, 4, figsize=(8, 2.8), sharey=True)
for ax, e in zip(axes, ENC):
    d = load(e, 'defense_mechanism.json')['variants']
    xx = np.arange(3)
    ax.bar(xx - 0.19, [1 - d[v]['adaptive v']['token_cos'] for v in VARS], 0.36, color=BLUE, label='encoder tokens')
    ax.bar(xx + 0.19, [1 - d[v]['adaptive v']['connector_cos'] for v in VARS], 0.36, color=ORANGE,
           label='connector output')
    ax.set_xticks(xx, ['no\ndefense', 'FT-\nfixed', 'FT-\nUAT'], fontsize=7.5); ax.set_title(NAMES[e], fontsize=9)
axes[0].set_ylabel('damage under the adaptive v\n(1 − cosine)'); axes[0].legend(fontsize=7.5)
plt.show()
'''),
md(r'''
Under the adaptive v, the defended encoders' **tokens** are clearly less damaged (SigLIP: 0.75 → about 0.5), but the damage at the **connector output**, which is what the language model reads, barely moves for SigLIP (0.34 → 0.31–0.33) and stays high for MAE. The adaptive attacker finds directions that the defended encoder lets through and that still reach the language model. SigLIP, where this gap is largest, is also where the defense is harmful in the captions. This is the mechanism behind the sentence of notebook 07, "robustness measured inside the encoder overstates robustness in the answers" (report IV.4, confidence B).

## Takeaways

| conf. | finding |
|---|---|
| A | Fine-tuning the encoder: negligible clean cost, effective against the fixed v, ineffective against the adaptive v, on all four encoders. |
| A | For SigLIP the defense is harmful against the adaptive attacker. |
| B | The defense protects the encoder tokens, not what the language model reads. |

**Limits.** One defense family, one training seed per defense, defense tuned and v built on the same 500 images (evaluations on images never used), adaptive attack with a single seed.
'''),
])

# =====================================================================================================================
NB10 = ('10_real_vlms_qwen_llava.ipynb', [
md(r'''
# Real VLMs: SmolVLM-256M, LLaVA-OneVision, Qwen2-VL

The mini-VLMs of notebook 08 are controlled but tiny. Here, the same protocol on VLMs trained by their authors:

| model | vision encoder (resolution) | language model |
|---|---|---|
| SmolVLM-256M (model of notebooks 01–07, splitting off) | SigLIP-B/16 (512) | SmolLM2-135M |
| LLaVA-OneVision-0.5B | SigLIP-SO400M (384) | Qwen2-0.5B |
| Qwen2-VL-2B | its own ViT (448) | Qwen2-1.5B |
| Qwen2-VL-2B at 224 px | same model, resolution of the mini-VLMs | |

Captions go through each model's own processor from 8-bit images; the attack's gradient goes through a differentiable copy of the preprocessing, checked against the processor. Two universal attacks: on the **encoder tokens** (as in notebook 01) and **end-to-end** on the clean caption likelihood.

Commands: `minivlm/run_real_vlm.sh <model>`, `run_endtoend.sh`, `run_single.sh` (per-image attack), `run_complete.sh` (seeds). Report: Parts VI and VII.
'''),
SETUP,
md(r'''
## 1. Overview
'''),
code(r'''
rows = []
for m in REAL:
    draws = [d for d in ['', ', draw 1', ', draw 2', ', draw 3', ', draw 4'] if has(m, 'random ±xi' + d)]
    noise = np.mean([inv2(m, 'random ±xi' + d)['rate'] for d in draws])
    seeds = [inv2(m, own(m, s))['rate'] for s in ['', 'seed 1', 'seed 2'] if has(m, own(m, s))]
    e2e = inv2(m, own(m, 'caption loss')) if has(m, own(m, 'caption loss')) else None
    rows.append([NAMES[m], pct(coco(m, 'clean', 'wrong'), False), '%.1f %% (%d draw%s)' % (noise, len(draws), 's' * (len(draws) > 1)),
                 pct(coco(m, own(m)), False) + ' / ' + pct(inv2(m, own(m)), False),
                 ' / '.join('%.1f' % s for s in seeds), pct(e2e) if e2e else 'not run (out of memory)'])
show_table(['model', 'clean, wrong (COCO)', 'random ±ξ, ImageNetV2', 'token v: COCO / ImageNetV2',
            'token v, ImageNetV2, seeds 0/1/2 (%)', 'end-to-end v, ImageNetV2'], rows, 'captions changed, ξ = 10/255')
'''),
md(r'''
SmolVLM-256M and LLaVA are clearly vulnerable, and their perturbed captions are false (high *wrong*). The perturbation of notebook 01 is reproduced here (report VI.1). Qwen2-VL looks much more robust under the token attack. The next sections show why that impression is wrong.

## 2. Qwen2-VL: the attack that almost concluded "robust"

A first end-to-end attack, with the step size used for the mini-VLMs (4/255), declared convergence after 23 epochs and changed only 7.8 % of the ImageNetV2 captions (report C.4; its captions are archived on the GPU machine, only its history is here). The same attack with a step eight times smaller:
'''),
code(r'''
fig, ax = plt.subplots(figsize=(7, 2.8))
for f, col, lab in [('old_lr4/uap_cap_xi10_history.json', GRAY, 'step 4/255 (first run)'),
                    ('uap_cap_xi10_history.json', BLUE, 'step 0.5/255')]:
    h = load('qwen2-vl-2b', f)
    ax.plot([r['epoch'] for r in h['history']], [r['build_score'] for r in h['history']], color=col, lw=2,
            label='%s: best %.3f at epoch %d' % (lab, h['best'], h['best_epoch']))
ax.set_xlabel('epoch'); ax.set_ylabel('likelihood of the clean caption\n(build images, lower = stronger)')
ax.legend(fontsize=8); plt.show()
print('ImageNetV2 captions changed with the 0.5/255 run:', pct(inv2('qwen2-vl-2b', own('qwen2-vl-2b', 'caption loss'))))
'''),
md(r'''
The well-optimised perturbation does more than change captions: it **injects a concept** into them.
'''),
code(r'''
import re
for m, pat, word in [('qwen2-vl-2b', 'mathemat|equation|formula', 'equations'),
                     ('qwen2-vl-2b-224', 'circuit|digital|futuristic|technolog', 'circuit / digital')]:
    rows = []
    for ds in ['coco', 'imagenetv2']:
        c = captions(m, ds)
        share = lambda caps: 100 * np.mean([bool(re.search(pat, s, re.I)) for s in caps])
        rows.append([ds, '%.1f %%' % share(c['clean']), '**%.1f %%**' % share(c[own(m, 'caption loss')])])
    show_table(['dataset', 'clean captions', 'under the end-to-end v'], rows,
               '%s: captions mentioning “%s”' % (NAMES[m], word))

c = captions('qwen2-vl-2b')
k = own('qwen2-vl-2b', 'caption loss')
shown = 0
for a, b in zip(c['clean'], c[k]):
    if re.search('equation', b, re.I) and len(a) < 110:
        print('clean:', a.strip()); print('  v  :', b.strip(), '\n'); shown += 1
    if shown == 4:
        break
'''),
md(r'''
Qwen2-VL still names the subject, but the attacker controls part of **every** caption. Our main metric counts only captions whose overall meaning changes, so it strongly undercounts this attack (report VI.2, confidence A). **Concluding robustness needs an attack checked to have gone all the way** (report VI.3).

## 3. Three seeds of the token attack on Qwen2-VL 448

Same attack, three starting points. The share of changed captions is recomputed from the captions, then compared seed against seed on the same images (McNemar).
'''),
code(r'''
m = 'qwen2-vl-2b'
STYLE = 'glitch|abstract|surreal|pixelat|distort'
keys = {s or 'seed 0': own(m, s) for s in ['', 'seed 1', 'seed 2']}
tok = load(m, 'uap_xi10_tokens.json')['token_cos']
flags, rows = {}, []
for ds in ['imagenetv2', 'coco']:
    c = captions(m, ds)
    for s, k in keys.items():
        flags[ds, s] = changed_flags(c['clean'], c[k])
for s, k in keys.items():
    style = [100 * np.mean([bool(re.search(STYLE, x, re.I)) for x in captions(m, ds)[k]]) for ds in ['coco', 'imagenetv2']]
    rows.append([s, '%.3f' % tok[k], '%.1f %%' % (100 * flags['coco', s].mean()),
                 '%.1f %%' % (100 * flags['imagenetv2', s].mean()), '%.1f %% / %.1f %%' % tuple(style)])
show_table(['seed', 'token cosine (held-out)', 'changed, COCO', 'changed, ImageNetV2',
            '“glitch / abstract / surreal …” style, COCO / ImageNetV2'], rows, 'Qwen2-VL 448, token v')
rows = []
for a, b in [('seed 0', 'seed 1'), ('seed 0', 'seed 2'), ('seed 1', 'seed 2')]:
    for ds in ['imagenetv2', 'coco']:
        x, y, p = mcnemar(flags[ds, a], flags[ds, b])
        rows.append(['%s vs %s' % (a, b), ds, '%d vs %d' % (x, y), '%.1g' % p])
show_table(['pair', 'dataset', 'discordant pairs', 'McNemar p'], rows)
'''),
code(r'''
c = captions(m)
for s in ['seed 1', 'seed 2']:
    shown = 0
    for a, b in zip(c['clean'], c[keys[s]]):
        if re.search(STYLE + '|digital', b, re.I) and len(b) < 130:
            print('%s | clean: %s\n       |   v  : %s\n' % (s, a.strip(), b.strip())); shown += 1
        if shown == 2:
            break
'''),
md(r'''
**Seed 0 is the outlier.** Seeds 1 and 2 each differ significantly from seed 0 and not from each other, and both inject a "glitch / abstract / digital" style that seed 0 never produces. A single seed, as in notebooks 01–07, would have under-estimated the vulnerability by a factor of about 2.5. The optimised score does not predict any of this: seed 2 has the best token cosine but injects the style less often than seed 1 (report VI.7, confidence B).

## 4. Transfers across resolutions are confounded by resizing
'''),
code(r'''
src = [('siglip', 224), ('clip', 224), ('ijepa', 224), ('mae', 224), ('llava-ov-0.5b', 384),
       ('smolvlm-256m', 512), ('qwen2-vl-2b', 448), ('qwen2-vl-2b-224', 224)]
fig, ax = plt.subplots(figsize=(8, 3.2))
for i, (s, res) in enumerate(src):
    for dx, tgt, col in [(-0.19, 'qwen2-vl-2b-224', BLUE), (0.19, 'qwen2-vl-2b', ORANGE)]:
        k = own(s) if s == tgt else 'universal v (%s), transfer' % s
        if has(tgt, k):
            bars(ax, [i + dx], [inv2(tgt, k)], col, None, 0.36)
for tgt, col in [('qwen2-vl-2b-224', BLUE), ('qwen2-vl-2b', ORANGE)]:
    ax.axhline(inv2(tgt, 'random ±xi')['rate'], color=col, lw=0.8, ls=':')
    ax.bar(0, 0, 0, color=col, label='attacked: %s' % NAMES[tgt])
ax.set_xticks(range(len(src)), ['%s\n(%d px)' % (NAMES[s].replace('-OV-0.5B', '').replace('Qwen2-VL ', 'Qwen '), r)
                                for s, r in src], fontsize=7.5)
ax.set_ylabel('ImageNetV2 captions changed (%)'); ax.legend(fontsize=8, ncol=2, loc='upper center')
ax.set_title('v built on ↓ (native resolution); dotted = noise ±ξ', fontsize=8.5, loc='left')
plt.show()
r = inv2('qwen2-vl-2b-224', 'universal v (qwen2-vl-2b), transfer')
print('Qwen 448 token v, downscaled to 224 px, on the SAME model at 224 px:', pct(r),
      '| noise:', pct(inv2('qwen2-vl-2b-224', 'random ±xi')))
'''),
md(r'''
A model's own perturbation, resized, falls to the noise level **on that same model**. Every transfer between models of different resolutions is therefore dominated by resizing, and says nothing about transferability. Only v built at 224 px fool Qwen at 224 px (report VI.5, confidence A).

## 5. Universal versus per-image perturbations

On 40 images, a perturbation optimised **for each image** (100 steps, likelihood of the clean caption), at three budgets, against the best universal v of each model.
'''),
code(r'''
BEST = {'smolvlm-256m': own('smolvlm-256m'), 'llava-ov-0.5b': own('llava-ov-0.5b'),
        'qwen2-vl-2b': own('qwen2-vl-2b', 'caption loss'), 'qwen2-vl-2b-224': own('qwen2-vl-2b-224', 'caption loss')}
rows = []
fig, axes = plt.subplots(1, 4, figsize=(8, 2.7), sharey=True)
for ax, m in zip(axes, REAL):
    r = load(m, 'single_image.json')['results']
    xs = [float(k.split()[1].split('/')[0]) for k in r]
    ys = [v['changed@0.5']['rate'] for v in r.values()]
    ax.plot(xs, ys, color=BLUE, lw=2, marker='o', label='per image (40 images)')
    bars(ax, [10], [inv2(m, BEST[m])], ORANGE, 'best universal v (500 images)', 0.9)
    ax.set_xticks([2, 4, 10]); ax.set_xlabel('ξ (/255)'); ax.set_title(NAMES[m], fontsize=9); ax.set_ylim(0, 100)
    rows.append([NAMES[m]] + ['%.1f %%' % y for y in ys] + [pct(inv2(m, BEST[m]), False)])
axes[0].set_ylabel('captions changed (%)'); axes[0].legend(fontsize=7, loc='lower right')
plt.show()
show_table(['model', 'per image 2/255', 'per image 4/255', 'per image 10/255', 'best universal v, 10/255'], rows)
'''),
md(r'''
Every model can be fooled image by image with a small budget. On Qwen2-VL the gap with the universal v is large in "changed", but this is not real robustness: its universal v injects a concept rather than replacing the caption, which the metric does not count (report VII.1, confidence B). With 40 images the intervals are about ±15 points.

## Takeaways

| conf. | finding |
|---|---|
| A | None of the real VLMs tested is robust to a universal perturbation run to convergence. |
| A | On Qwen2-VL the perturbation injects a concept ("mathematical equations") into most captions; "changed" undercounts it. |
| A | A first, poorly optimised attack nearly concluded "Qwen2-VL is robust". |
| A | Transfers between different resolutions are confounded with resizing. |
| B | On Qwen2-VL 448 the attack seed changes the result and the mode of attack; seed 0 was the outlier. |

**Missing.** End-to-end attack and seeds on LLaVA (ran out of GPU memory; `./run_complete.sh llava`).
'''),
])

# =====================================================================================================================
NB11 = ('11_injection_scan_and_imagenet_accuracy.ipynb', [
md(r'''
# Injected words, and ImageNet accuracy read from the captions

Notebook 10 found the "mathematical equations" injection by reading captions. To avoid depending on what we happened to read, this notebook **searches automatically**: for each model's own perturbation, the words present in **at most 1 %** of clean captions and **at least 10 %** of perturbed captions (same images, stop words removed). The thresholds were set before looking at the results. Transfers and random noise are excluded.

The search is `injection_scan.scan`, run here live on the saved captions (seconds). Report: Parts VIII and IX.
'''),
SETUP,
code(r'''
from injection_scan import scan, MODELS, CLEAN_MAX, PERT_MIN
res = {m: {ds: scan(captions(m, ds)) for ds in ['imagenetv2', 'coco']} for m in MODELS}
LEXICAL_NOISE = {('smolvlm-256m', own('smolvlm-256m')), ('smolvlm-256m', own('smolvlm-256m', 'seed 1'))}   # sorted by hand
rows, n_none, n_noise, n_total = [], 0, 0, 0
for m in MODELS:
    for k, r in res[m]['imagenetv2'].items():
        rc = res[m]['coco'].get(k, {'any_injected_word': float('nan')})
        words = ', '.join('%s %.0f%%' % (w, 100 * x) for w, x in list(r['words'].items())[:5])
        rows.append([NAMES[m], k.replace('universal v (%s' % m, '').strip('(), ') or 'token v, seed 0',
                     '%.0f %%' % (100 * r['any_injected_word']), '%.0f %%' % (100 * rc['any_injected_word']), words or '—'])
        n_total += 1; n_none += not r['words']; n_noise += (m, k) in LEXICAL_NOISE
show_table(['model', 'perturbation', 'captions with ≥ 1 injected word, ImageNetV2', 'same, COCO',
            'injected words (share of ImageNetV2 captions)'], rows)
print('%d own perturbations: %d inject no word, %d only lexical noise (SmolVLM token v, seeds 0 and 1), '
      '%d make foreign words appear in >= %d %% of ImageNetV2 captions'
      % (n_total, n_none, n_noise, n_total - n_none - n_noise, 100 * PERT_MIN))
saved = json.load(open(RESULTS_DIR / 'injection_scan.json'))
print('same result as results/injection_scan.json:', saved == json.loads(json.dumps(res)))
'''),
md(r'''
**Injection is the general case, not a peculiarity of Qwen** (report IX.1, confidence B). Three modes appear:
- **replacement** by a dominant caption (MAE token v: "…next to each other");
- **concept** added to the content or partly replacing it (LLaVA "baseball team", Qwen "equations", I-JEPA "book");
- **style** that leaves the content intact (Qwen "digital / glitch / abstract").

The injected word is sometimes stable across seeds ("book" for I-JEPA, "digital" for Qwen 224, "picture" for SigLIP), sometimes not (Qwen 448, CLIP). Some found words are lexical noise (SmolVLM's "o", "again"); the sorting into modes is done by hand.
'''),
code(r'''
import re
EXAMPLES = [('llava-ov-0.5b', own('llava-ov-0.5b'), 'baseball'), ('ijepa', own('ijepa', 'seed 2'), 'book'),
            ('qwen2-vl-2b-224', own('qwen2-vl-2b-224'), 'digital'), ('mae', own('mae', 'caption loss'), 'bathroom'),
            ('siglip', own('siglip'), 'picture')]
for m, k, w in EXAMPLES:
    c = captions(m)
    for a, b in zip(c['clean'], c[k]):
        if re.search(r'\b%s\b' % w, b, re.I) and len(b) < 120:
            print('%-15s “%s”\n%-15s → “%s”\n' % (NAMES[m], a.strip(), '', b.strip()))
            break
'''),
md(r'''
## Concept and style injected into Qwen2-VL
'''),
code(r'''
INJ = [('qwen2-vl-2b', own('qwen2-vl-2b', 'caption loss'), 'mathemat|equation|formula', '"mathematical equations"', BLUE),
       ('qwen2-vl-2b-224', own('qwen2-vl-2b-224', 'caption loss'), 'circuit|digital|futuristic|technolog', '"circuit / digital"', BLUE),
       ('qwen2-vl-2b', own('qwen2-vl-2b', 'seed 1'), 'glitch|abstract|surreal|pixelat|distort', '"glitch / abstract" style', AQUA),
       ('qwen2-vl-2b', own('qwen2-vl-2b', 'seed 2'), 'glitch|abstract|surreal|pixelat|distort', '"glitch / abstract" style', AQUA)]
fig, ax = plt.subplots(figsize=(7, 4.6))
labels, y = [], 0
for m, k, pat, word, col in INJ:
    for ds in ['coco', 'imagenetv2']:
        c = captions(m, ds)
        share = lambda caps: 100 * np.mean([bool(re.search(pat, s, re.I)) for s in caps])
        ax.barh(y + 0.18, share(c[k]), 0.34, color=col); ax.barh(y - 0.18, share(c['clean']), 0.34, color=GRAY)
        ax.text(share(c[k]) + 1, y + 0.18, '%.0f %%' % share(c[k]), va='center', fontsize=7.5)
        labels.append('%s, %s, %s\n%s' % (NAMES[m], k.split('), ')[-1] if 'seed' in k else 'end-to-end v', ds, word))
        y -= 1
ax.set_yticks(np.arange(0, y, -1), labels, fontsize=7); ax.set_xlim(0, 100)
ax.set_xlabel('captions mentioning the concept (%)'); ax.grid(axis='x', color=GRID); ax.grid(axis='y', visible=False)
ax.barh(0, 0, 0, color=BLUE, label='end-to-end v'); ax.barh(0, 0, 0, color=AQUA, label='token v')
ax.barh(0, 0, 0, color=GRAY, label='clean'); ax.legend(fontsize=7.5, loc='lower right')
plt.show()
'''),
md(r'''
## ImageNet accuracy read from the captions

A metric independent of the 0.5 threshold: each caption of the 500 held-out images is matched (MiniLM) against "a photo of a <class>" for the 1,000 ImageNet classes; top-1 against the true class. Computed by `python imagenet_accuracy.py` (reads the saved captions). Captions are short and generic, so absolute values are low; compare conditions within a model, not with a classifier.
'''),
code(r'''
acc = json.load(open(RESULTS_DIR / 'imagenet_accuracy.json'))
rows = []
for m in ENC + REAL:
    a = acc[m]
    get = lambda k: '%.1f %%' % a[k]['top1'] if k in a else '—'
    rows.append([NAMES[m], get('clean'), get(own(m)), get(own(m, 'caption loss')), get('random ±xi')])
show_table(['model', 'clean', 'token v (seed 0)', 'end-to-end v', 'noise ±ξ'], rows, 'top-1 accuracy, ImageNetV2 held-out (±4 points)')
'''),
md(r'''
The ranking of the models is the same as with "changed". For Qwen2-VL the class is still often named under attack while a foreign concept or style is added, consistent with injection (report VIII.1).

**Limits.** Word-level search: no paraphrases, no multi-word concepts; arbitrary (but pre-set) thresholds; mode classification by hand.
'''),
])

# =====================================================================================================================
NB12 = ('12_live_demo_smolvlm.ipynb', [
md(r'''
# Live demo: a saved universal perturbation on SmolVLM-256M

The only notebook of the series that runs a model. It loads **SmolVLM-256M** (the model of notebooks 01–07) and the universal perturbation built for it by `real_vlm.py` (token attack, run to convergence, ξ = 10/255, 512 × 512). It then captions a few images that **were never used to build v**: clean, with v, and with random noise of the same size.

Runs on CPU in a few minutes (no GPU needed). It needs the Hugging Face cache of `HuggingFaceTB/SmolVLM-256M-Instruct` and `sentence-transformers/all-MiniLM-L6-v2`, plus a folder of images (default: the Imagenette validation set used in `python/`). Optional: the same with Qwen2-VL-2B and its end-to-end v (slower, set `RUN_QWEN = True`).
'''),
code(r'''
import os, sys, glob
sys.path.insert(0, os.path.abspath('minivlm'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')         # use the local cache only
import numpy as np, torch
import matplotlib.pyplot as plt
from nb_utils import *
from real_vlm import RealVLM, load_square, load_any_v
from common import XI
use_style()

IMAGE_GLOB = '../python/data/imagenette2-320/val/*/*.JPEG'   # any folder of photos works
N_IMAGES = 6
DEVICE = 'cuda' if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 4e9 else 'cpu'
RUN_QWEN = False
torch.manual_seed(0)
files = sorted(glob.glob(IMAGE_GLOB))
files = files[::max(1, len(files) // N_IMAGES)][:N_IMAGES]   # spread over the classes
print(len(files), 'images, device', DEVICE)
'''),
code(r'''
vlm = RealVLM('smolvlm-256m', DEVICE)
u8 = load_square(files, vlm.size)                               # 8-bit images, 512 x 512
v = load_any_v(RESULTS_DIR / 'smolvlm-256m' / 'uap_xi10.pt', vlm.size, XI, DEVICE)
noise = (torch.randint(0, 2, v.shape, device=DEVICE).float() * 2 - 1) * XI   # random ±ξ, same l_inf size
print('l_inf of v: %.1f/255, of the noise: %.1f/255' % (255 * v.abs().max(), 255 * noise.abs().max()))
'''),
code(r'''
perturbed = torch.round(((u8.float() / 255 + v.cpu()).clamp(0, 1)) * 255).to(torch.uint8)
fig, axes = plt.subplots(2, 4, figsize=(8, 4.3))
for i in range(3):
    axes[0, i].imshow(u8[i].permute(1, 2, 0)); axes[0, i].set_title('clean', fontsize=8)
    axes[1, i].imshow(perturbed[i].permute(1, 2, 0)); axes[1, i].set_title('with v', fontsize=8)
axes[0, 3].imshow(v_image(v[0].cpu())); axes[0, 3].set_title('v (contrast stretched)', fontsize=8)
axes[1, 3].imshow(v_image(noise[0].cpu())); axes[1, 3].set_title('random ±ξ', fontsize=8)
for a in axes.flat:
    a.axis('off')
plt.show()
'''),
code(r'''
caps = {'clean': vlm.caption(u8, bs=3), 'universal v': vlm.caption(u8, v, bs=3), 'random ±ξ': vlm.caption(u8, noise, bs=3)}
sim_v, sim_n = caption_sim(caps['clean'], caps['universal v']), caption_sim(caps['clean'], caps['random ±ξ'])
rows = [[os.path.basename(os.path.dirname(f)), caps['clean'][i], '%s (%.2f)' % (caps['universal v'][i], sim_v[i]),
         '%s (%.2f)' % (caps['random ±ξ'][i], sim_n[i])] for i, f in enumerate(files)]
show_table(['image (class folder)', 'clean caption', 'with the universal v (similarity)', 'with random noise (similarity)'], rows)
print('changed (similarity < 0.5): universal v %d / %d, noise %d / %d' % ((sim_v < 0.5).sum(), len(files), (sim_n < 0.5).sum(), len(files)))
'''),
md(r'''
The perturbation is barely visible, the same for every image, and was built on other images; random noise of the same size leaves the captions essentially intact.

### Optional: Qwen2-VL-2B and the "mathematical equations" injection

Set `RUN_QWEN = True` above. Qwen2-VL-2B at 448 px with its end-to-end universal v (notebook 10, section 2). On CPU, expect about a minute per caption.
'''),
code(r'''
if RUN_QWEN:
    del vlm
    q = RealVLM('qwen2-vl-2b', DEVICE)
    uq = load_square(files[:3], q.size)
    vq = load_any_v(RESULTS_DIR / 'qwen2-vl-2b' / 'uap_cap_xi10.pt', q.size, XI, DEVICE)
    for a, b in zip(q.caption(uq, bs=1), q.caption(uq, vq, bs=1)):
        print('clean:', a, '\n  v  :', b, '\n')
else:
    print('skipped (RUN_QWEN = False)')
'''),
md(r'''
**To rebuild a perturbation** rather than load it (GPU, about an hour for SmolVLM-256M):

```
cd minivlm
python real_vlm.py --model smolvlm-256m                 # token attack, to convergence, then evaluation
python real_vlm.py --model smolvlm-256m --target caption --lr 1   # end-to-end attack
```
'''),
])

NOTEBOOKS = {'08': NB08, '09': NB09, '10': NB10, '11': NB11, '12': NB12}


# =====================================================================================================================
def figure_outputs():
    outs = []
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format='png', bbox_inches='tight', dpi=110)
        outs.append({'output_type': 'display_data', 'metadata': {},
                     'data': {'image/png': base64.b64encode(buf.getvalue()).decode(), 'text/plain': ['<Figure>']}})
    plt.close('all')
    return outs


def execute(cells):
    import nb_utils
    os.chdir(NB_DIR)
    ns = {'__name__': '__main__'}
    results, current = [], []
    nb_utils.DISPLAY = lambda m: current.append({'output_type': 'display_data', 'metadata': {},
                                                 'data': {'text/markdown': m, 'text/plain': m}})
    for i, (kind, src) in enumerate(cells):
        current.clear()
        if kind != 'code':
            results.append(None)
            continue
        buf = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
            warnings.simplefilter('ignore')
            try:
                exec(compile(src, '<cell %d>' % i, 'exec'), ns)
            except Exception:
                sys.stderr.write(buf.getvalue())
                raise
        outs = []
        if buf.getvalue():
            outs.append({'output_type': 'stream', 'name': 'stdout', 'text': buf.getvalue()})
        # Markdown tables and figures in the order they were produced is not tracked: text, tables, then figures
        outs += list(current) + figure_outputs()
        results.append(outs)
    return results


def write(name, cells, outputs):
    nb_cells, count = [], 0
    for i, (kind, src) in enumerate(cells):
        lines = src.splitlines(keepends=True)
        if kind == 'markdown':
            nb_cells.append({'cell_type': 'markdown', 'id': 'c%02d' % i, 'metadata': {}, 'source': lines})
        else:
            count += 1
            nb_cells.append({'cell_type': 'code', 'id': 'c%02d' % i, 'metadata': {}, 'source': lines,
                             'execution_count': count if outputs else None,
                             'outputs': (outputs[i] or []) if outputs else []})
    nb = {'cells': nb_cells, 'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                       'language_info': {'name': 'python'}}}
    json.dump(nb, open(NB_DIR / name, 'w'), indent=1, ensure_ascii=False)
    print('>>', NB_DIR / name)


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    for key in args or list(NOTEBOOKS):
        name, cells = NOTEBOOKS[key]
        try:
            outputs = None if '--no-exec' in sys.argv else execute(cells)
        except Exception:
            traceback.print_exc()
            print('!! %s failed, written without outputs' % name)
            outputs = None
        write(name, cells, outputs)
