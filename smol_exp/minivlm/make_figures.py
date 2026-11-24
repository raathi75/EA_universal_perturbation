"""Figures of RAPPORT.md / REPORT_EN.md, read directly from the result files (nothing typed by hand)
  python make_figures.py              -> results/figures/*.png     (French labels)
  python make_figures.py --lang en    -> results/figures_en/*.png  (English labels)
"""
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import RESULTS_DIR

EN = '--lang' in sys.argv and sys.argv[sys.argv.index('--lang') + 1] == 'en'
OUT = RESULTS_DIR / ('figures_en' if EN else 'figures')


def L(fr, en):
    return en if EN else fr


OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.axisbelow': True, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.dpi': 200,
                     'legend.frameon': False})
ENC = ['siglip', 'clip', 'ijepa', 'mae']
NAMES = {'siglip': 'SigLIP', 'clip': 'CLIP', 'ijepa': 'I-JEPA', 'mae': 'MAE', 'smolvlm-256m': 'SmolVLM-256M',
         'llava-ov-0.5b': 'LLaVA-OV-0.5B', 'qwen2-vl-2b': 'Qwen2-VL 448', 'qwen2-vl-2b-224': 'Qwen2-VL 224'}
REAL = ['smolvlm-256m', 'llava-ov-0.5b', 'qwen2-vl-2b', 'qwen2-vl-2b-224']


def load(m, f):
    return json.load(open(RESULTS_DIR / m / f))


def coco(m, k):
    return load(m, 'coco_stage2.json')['results'][k]['changed']


def inv2(m, k):
    return load(m, 'eval_captions_stage2_xi10.json')['results'][k]['changed@0.5']['all']


def err(r):
    return [[r['rate'] - r['ci95'][0]], [r['ci95'][1] - r['rate']]]


def bars(ax, x, vals, color, label, w):
    for xi, r in zip(x, vals):
        ax.bar(xi, r['rate'], w, color=color, edgecolor=SURF, linewidth=1)
        ax.errorbar(xi, r['rate'], yerr=err(r), color=INK, lw=0.8, capsize=2)
    ax.bar(0, 0, 0, color=color, label=label)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print('>>', OUT / name)


# 1. vulnerability of the mini-VLMs -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.0))
x = np.arange(len(ENC))
bars(ax, x - 0.27, [coco(e, 'universal v (%s)' % e) for e in ENC], BLUE, L('v propre, COCO val', 'own v, COCO val'), 0.26)
bars(ax, x, [inv2(e, 'universal v (%s)' % e) for e in ENC], ORANGE, L('v propre, ImageNetV2 held-out', 'own v, ImageNetV2 held-out'), 0.26)
bars(ax, x + 0.27, [inv2(e, 'random ±xi') for e in ENC], GRAY, L('bruit ±ξ, ImageNetV2', 'noise ±ξ, ImageNetV2'), 0.26)
ax.set_xticks(x, [NAMES[e] for e in ENC])
ax.set_ylabel(L('légendes changées (%)', 'captions changed (%)'))
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=8)
ax.set_title(L('Mini-VLM, même recette, ξ = 10/255 (IC 95 % de Wilson)', 'Mini-VLMs, same recipe, ξ = 10/255 (95 % Wilson CI)'), fontsize=9, color=INK, loc='left')
save(fig, 'fig1_minivlm.png')

# 2. dominant caption -----------------------------------------------------------------------------------------
models = ENC + ['smolvlm-256m', 'llava-ov-0.5b', 'qwen2-vl-2b', 'qwen2-vl-2b-224']
top_v, top_c = [], []
for m in models:
    c = load(m, 'coco_captions_stage2.json')
    top_v.append(Counter(c['universal v (%s)' % m]).most_common(1)[0][1])
    top_c.append(Counter(c['clean']).most_common(1)[0][1])
fig, ax = plt.subplots(figsize=(6.4, 2.9))
y = np.arange(len(models))[::-1]
ax.barh(y + 0.18, top_v, 0.34, color=BLUE, label=L('sous la perturbation universelle propre', 'under its own universal perturbation'), edgecolor=SURF)
ax.barh(y - 0.18, top_c, 0.34, color=GRAY, label=L('images propres', 'clean images'), edgecolor=SURF)
for yi, t in zip(y, top_v):
    ax.text(t + 8, yi + 0.18, str(t), va='center', fontsize=7.5, color=INK2)
ax.set_yticks(y, [NAMES[m] for m in models])
ax.set_xlabel(L('nombre d\'images COCO (sur 1000) recevant la légende la plus fréquente', 'COCO images (out of 1000) given the most frequent caption'))
ax.grid(axis='x', color=GRID)
ax.grid(axis='y', visible=False)
ax.set_xlim(0, 1000)
ax.legend(loc='lower right', fontsize=8)
save(fig, 'fig2_dominant_caption.png')

# 3. defense ---------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.7), sharey=True)
for ax, e in zip(axes, ENC):
    d = load(e, 'defense_ft.json')['variants']
    var = ['no-defense', 'FT-fixed', 'FT-UAT']
    xx = np.arange(3)
    fixed = [d[v]['imagenetv2']['fixed v']['changed@0.5']['all'] if v != 'no-defense' else inv2(e, 'universal v (%s)' % e)
             for v in var]
    adapt = [d[v]['imagenetv2']['adaptive v']['changed@0.5']['all'] for v in var]
    bars(ax, xx - 0.19, fixed, BLUE, L('v fixe', 'fixed v'), 0.36)
    bars(ax, xx + 0.19, adapt, ORANGE, L('v adaptatif', 'adaptive v'), 0.36)
    ax.set_xticks(xx, [L('sans\ndéfense', 'no\ndefense'), 'FT-\nfixed', 'FT-\nUAT'], fontsize=7.5)
    ax.set_title(NAMES[e], fontsize=9, color=INK)
    ax.set_ylim(0, 100)
axes[0].set_ylabel(L('légendes ImageNetV2 changées (%)', 'ImageNetV2 captions changed (%)'))
axes[0].legend(loc='upper left', fontsize=7.5)
save(fig, 'fig3_defense.png')

# 4. per-image vs universal ------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5), sharey=True)
best_univ = {'smolvlm-256m': 'universal v (smolvlm-256m)', 'llava-ov-0.5b': 'universal v (llava-ov-0.5b)',
             'qwen2-vl-2b': 'universal v (qwen2-vl-2b, caption loss)',
             'qwen2-vl-2b-224': 'universal v (qwen2-vl-2b-224, caption loss)'}
for ax, m in zip(axes, REAL):
    r = load(m, 'single_image.json')['results']
    xs = [float(k.split()[1].split('/')[0]) for k in r]
    ys = [v['changed@0.5']['rate'] for v in r.values()]
    ax.plot(xs, ys, color=BLUE, lw=2, marker='o', ms=5, label=L('une perturbation par image (40 images)', 'one perturbation per image (40 images)'))
    u = inv2(m, best_univ[m])
    ax.plot([10], [u['rate']], marker='s', ms=7, color=ORANGE, ls='none', label=L('meilleur v universel (500 images)', 'best universal v (500 images)'))
    ax.errorbar([10], [u['rate']], yerr=err(u), color=INK, lw=0.8, capsize=2)
    ax.set_xticks([2, 4, 10])
    ax.set_xlabel('ξ (/255)')
    ax.set_title(NAMES[m], fontsize=9, color=INK)
    ax.set_ylim(0, 100)
axes[0].set_ylabel(L('légendes changées (%)', 'captions changed (%)'))
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / 'fig4_per_image_vs_universal.png')
plt.close(fig)
print('>>', OUT / 'fig4_per_image_vs_universal.png')

# 5. resolution confound: transfers onto Qwen2-VL at 224 and 448 ----------------------------------------------
src = [('siglip', 224), ('clip', 224), ('ijepa', 224), ('mae', 224), ('llava-ov-0.5b', 384),
       ('smolvlm-256m', 512), ('qwen2-vl-2b', 448), ('qwen2-vl-2b-224', 224)]
fig, ax = plt.subplots(figsize=(7.0, 3.1))
x = np.arange(len(src))
for i, (s, res) in enumerate(src):
    for dx, tgt, col in [(-0.19, 'qwen2-vl-2b-224', BLUE), (0.19, 'qwen2-vl-2b', ORANGE)]:
        k = 'universal v (%s)' % s if s == tgt else 'universal v (%s), transfer' % s
        rr = load(tgt, 'eval_captions_stage2_xi10.json')['results'].get(k)
        if rr:
            r = rr['changed@0.5']['all']
            ax.bar(i + dx, r['rate'], 0.36, color=col, edgecolor=SURF)
            ax.errorbar(i + dx, r['rate'], yerr=err(r), color=INK, lw=0.8, capsize=2)
ax.bar(0, 0, 0, color=BLUE, label=L('attaqué : Qwen2-VL à 224 px', 'attacked: Qwen2-VL at 224 px'))
ax.bar(0, 0, 0, color=ORANGE, label=L('attaqué : Qwen2-VL à 448 px', 'attacked: Qwen2-VL at 448 px'))
for lvl, col in [(inv2('qwen2-vl-2b-224', 'random ±xi')['rate'], BLUE), (inv2('qwen2-vl-2b', 'random ±xi')['rate'], ORANGE)]:
    ax.axhline(lvl, color=col, lw=0.8, ls=':')
ax.set_xticks(x, ['%s\n(%d px)' % (NAMES.get(s, s).replace('-OV-0.5B', '').replace('-256M', '').replace('Qwen2-VL ', 'Qwen '), r)
                  for s, r in src], fontsize=7.5)
ax.set_ylim(0, 32)
ax.set_ylabel(L('légendes ImageNetV2 changées (%)', 'ImageNetV2 captions changed (%)'))
ax.legend(loc='upper center', fontsize=8, ncol=2)
ax.set_title(L('v construit sur ↓ (résolution native) ; pointillés = bruit ±ξ', 'v built on ↓ (native resolution); dotted = noise ±ξ'), fontsize=8.5, color=INK2, loc='left')
save(fig, 'fig5_resolution_transfer.png')

# 6. convergence of the token attacks ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, group, title in [(axes[0], ENC, 'mini-VLM'), (axes[1], REAL, L('VLM réels', 'real VLMs'))]:
    cols = [BLUE, ORANGE, AQUA, '#eda100']
    for m, col in zip(group, cols):
        h = load(m, 'uap_xi10_history.json')['history']
        ax.plot([r['epoch'] for r in h], [r['build_score'] for r in h], color=col, lw=2, label=NAMES[m])
    ax.set_xlabel(L('époque', 'epoch'))
    ax.set_title(title, fontsize=9, color=INK)
    ax.legend(fontsize=7.5)
axes[0].set_ylabel(L('cosinus des tokens (images build)', 'token cosine (build images)'))
save(fig, 'fig6_convergence.png')

# 7. the perturbations themselves ------------------------------------------------------------------------------
show = ENC + REAL
fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.3), gridspec_kw={'hspace': 0.3})
for ax, m in zip(axes.flat, show):
    v = torch.load(RESULTS_DIR / m / 'uap_xi10.pt', map_location='cpu').float().reshape(3, *torch.load(
        RESULTS_DIR / m / 'uap_xi10.pt', map_location='cpu').shape[-2:])
    img = (v / (2 * v.abs().max()) + 0.5).permute(1, 2, 0).numpy()
    ax.imshow(img, interpolation='nearest')
    ax.set_title('%s (%d px)' % (NAMES[m], v.shape[-1]), fontsize=8, color=INK)
    ax.axis('off')
fig.suptitle(L('Perturbations universelles propres (v tokens), contraste étiré : gris = 0, ±ξ = saturé', 'Own universal perturbations (token v), stretched contrast: gray = 0, ±ξ = saturated'), fontsize=8.5,
             color=INK2)
fig.savefig(OUT / 'fig7_perturbations.png', bbox_inches='tight')
plt.close(fig)

# 8. concept injected by universal perturbations of Qwen2-VL ------------------------------------------------------
import re
inj = [('qwen2-vl-2b', 'universal v (qwen2-vl-2b, caption loss)', 'mathemat|equation|formula',
        L('« mathematical equations »', '"mathematical equations"'), L('v bout en bout', 'end-to-end v')),
       ('qwen2-vl-2b-224', 'universal v (qwen2-vl-2b-224, caption loss)', 'circuit|digital|futuristic|technolog',
        L('« circuit / digital / futuristic »', '"circuit / digital / futuristic"'), L('v bout en bout', 'end-to-end v')),
       ('qwen2-vl-2b', 'universal v (qwen2-vl-2b), seed 1', 'glitch|abstract|surreal|pixelat|distort',
        L('« glitch / abstract / surreal »', '"glitch / abstract / surreal"'), L('v tokens, graine 1', 'token v, seed 1')),
       ('qwen2-vl-2b', 'universal v (qwen2-vl-2b), seed 2', 'glitch|abstract|surreal|pixelat|distort',
        L('« glitch / abstract / surreal »', '"glitch / abstract / surreal"'), L('v tokens, graine 2', 'token v, seed 2'))]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
labels, yy = [], 0
for m, k, pat, word, which in inj:
    for f, ds in [('coco_captions_stage2.json', 'COCO val'), ('captions_stage2_xi10.json', 'ImageNetV2')]:
        c = load(m, f)
        if k not in c:
            continue
        n = lambda caps: 100 * sum(bool(re.search(pat, x, re.I)) for x in caps) / len(caps)
        col = AQUA if 'seed' in k else BLUE
        ax.barh(yy + 0.18, n(c[k]), 0.34, color=col, edgecolor=SURF)
        ax.barh(yy - 0.18, n(c['clean']), 0.34, color=GRAY, edgecolor=SURF)
        ax.text(n(c[k]) + 1, yy + 0.18, '%.0f %%' % n(c[k]), va='center', fontsize=7.5, color=INK2)
        labels.append('%s, %s, %s\n%s' % (NAMES[m], which, ds, word))
        yy -= 1
ax.barh(0, 0, 0, color=BLUE, label=L('avec le v universel de bout en bout', 'with the end-to-end universal v'))
ax.barh(0, 0, 0, color=AQUA, label=L('avec le v tokens (graines 1 et 2)', 'with the token v (seeds 1 and 2)'))
ax.barh(0, 0, 0, color=GRAY, label=L('images propres', 'clean images'))
ax.set_yticks(np.arange(0, yy, -1), labels, fontsize=7)
ax.set_xlim(0, 100)
ax.set_xlabel(L('légendes mentionnant le concept (%)', 'captions mentioning the concept (%)'))
ax.grid(axis='x', color=GRID)
ax.grid(axis='y', visible=False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.2)
save(fig, 'fig8_concept_injection.png')

# 9. defense mechanism: token vs connector-output damage under the adaptive v ------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6), sharey=True)
for ax, e in zip(axes, ENC):
    d = load(e, 'defense_mechanism.json')['variants']
    var = ['no-defense', 'FT-fixed', 'FT-UAT']
    xx = np.arange(3)
    ax.bar(xx - 0.19, [1 - d[v]['adaptive v']['token_cos'] for v in var], 0.36, color=BLUE, edgecolor=SURF,
           label=L('tokens de l\'encodeur', 'encoder tokens'))
    ax.bar(xx + 0.19, [1 - d[v]['adaptive v']['connector_cos'] for v in var], 0.36, color=ORANGE, edgecolor=SURF,
           label=L('sorties du connecteur', 'connector outputs'))
    ax.set_xticks(xx, [L('sans\ndéfense', 'no\ndefense'), 'FT-\nfixed', 'FT-\nUAT'], fontsize=7.5)
    ax.set_title(NAMES[e], fontsize=9, color=INK)
    ax.set_ylim(0, 1)
axes[0].set_ylabel(L('dégât sous v adaptatif (1 − cosinus)', 'damage under adaptive v (1 − cosine)'))
axes[0].legend(loc='upper right', fontsize=7)
save(fig, 'fig9_defense_mechanism.png')
