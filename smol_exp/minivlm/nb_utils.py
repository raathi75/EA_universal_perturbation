"""Helpers of the result notebooks 08-12 (smol_exp/*.ipynb).

The notebooks READ the result files written by the experiment scripts (results/<model>/*.json, *.pt); they do not
rerun the attacks, which take hours of GPU time. What they recompute is cheap and checkable: statistics (Wilson
intervals, McNemar tests), caption similarities (MiniLM, the metric of the report) and the injected-word search.
Every number of RAPPORT.md / REPORT_EN.md can thus be traced to a cell.
"""
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import CAPTION_SIM_CHANGED, RESULTS_DIR

ENC = ['siglip', 'clip', 'ijepa', 'mae']
REAL = ['smolvlm-256m', 'llava-ov-0.5b', 'qwen2-vl-2b', 'qwen2-vl-2b-224']
NAMES = {'siglip': 'SigLIP', 'clip': 'CLIP', 'ijepa': 'I-JEPA', 'mae': 'MAE', 'smolvlm-256m': 'SmolVLM-256M',
         'llava-ov-0.5b': 'LLaVA-OV-0.5B', 'qwen2-vl-2b': 'Qwen2-VL 448', 'qwen2-vl-2b-224': 'Qwen2-VL 224',
         'siglip-unfrozen': 'SigLIP (unfrozen)'}
SIZE = {'siglip': 224, 'clip': 224, 'ijepa': 224, 'mae': 224, 'smolvlm-256m': 512, 'llava-ov-0.5b': 384,
        'qwen2-vl-2b': 448, 'qwen2-vl-2b-224': 224}

BLUE, ORANGE, AQUA, YELLOW, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#9a9993'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'


def use_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2,
                         'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'figure.dpi': 110, 'legend.frameon': False})


# ---------------------------------------------------------------------------------------------- result files
def load(m, f):
    return json.load(open(RESULTS_DIR / m / f))


def own(m, variant=''):
    """Key of a model's own universal v. variant: '' (token v, seed 0), 'seed 1', 'seed 2', 'caption loss',
    'connector-output'."""
    if variant.startswith('seed'):
        return 'universal v (%s), %s' % (m, variant)
    return 'universal v (%s%s)' % (m, ', ' + variant if variant else '')


def inv2(m, k, threshold=0.5):
    """ImageNetV2 held-out (500 images): share of captions changed, {'rate', 'ci95', 'changed', 'n'}."""
    return load(m, 'eval_captions_stage2_xi10.json')['results'][k]['changed@%s' % threshold]['all']


def coco(m, k, what='changed'):
    """COCO val (1000 images): 'changed' (vs the clean caption) or 'wrong' (vs the 5 human captions)."""
    return load(m, 'coco_stage2.json')['results'][k][what]


def captions(m, dataset='imagenetv2'):
    return load(m, 'captions_stage2_xi10.json' if dataset == 'imagenetv2' else 'coco_captions_stage2.json')


def has(m, k, dataset='imagenetv2'):
    f = 'eval_captions_stage2_xi10.json' if dataset == 'imagenetv2' else 'coco_stage2.json'
    p = RESULTS_DIR / m / f
    return p.exists() and k in json.load(open(p))['results']


def pct(r, ci=True):
    if r is None:
        return '—'
    return '%.1f %% (%.1f–%.1f)' % (r['rate'], *r['ci95']) if ci else '%.1f %%' % r['rate']


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return 100 * (c - h), 100 * (c + h)


# ---------------------------------------------------------------------------------------------- display
DISPLAY = None   # set by the notebook builder to capture tables as Markdown outputs


def show_table(headers, rows, title=None):
    md = ('**%s**\n\n' % title if title else '') + '| ' + ' | '.join(headers) + ' |\n|' + '---|' * len(headers) + '\n'
    md += '\n'.join('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows)
    if DISPLAY is not None:
        DISPLAY(md)
        return
    try:
        from IPython.display import Markdown, display
        display(Markdown(md))
    except ImportError:
        print(md)


# ---------------------------------------------------------------------------------------------- statistics
_SIM = None


def caption_sim(a, b):
    """MiniLM cosine between two lists of captions (the 'changed' metric of the report, threshold 0.5)."""
    global _SIM
    if _SIM is None:
        from common import TextSim
        _SIM = TextSim('cuda' if torch.cuda.is_available() else 'cpu')
    return _SIM(list(a), list(b)).numpy()


def changed_flags(clean, pert, threshold=CAPTION_SIM_CHANGED):
    return caption_sim(clean, pert) < threshold


def mcnemar(flags_a, flags_b):
    """Exact McNemar test on paired binary outcomes: (only a, only b, two-sided p)."""
    from scipy.stats import binomtest
    a, b = np.asarray(flags_a, bool), np.asarray(flags_b, bool)
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return x, y, (binomtest(x, x + y).pvalue if x + y else 1.0)


def dominant(caps):
    """(count, caption) of the most frequent caption."""
    c, n = Counter(caps).most_common(1)[0]
    return n, c


# ---------------------------------------------------------------------------------------------- plotting
def bars(ax, x, vals, color, label, w):
    for xi, r in zip(x, vals):
        if r is None:
            continue
        ax.bar(xi, r['rate'], w, color=color, edgecolor=SURF, linewidth=1)
        ax.errorbar(xi, r['rate'], yerr=[[r['rate'] - r['ci95'][0]], [r['ci95'][1] - r['rate']]], color=INK, lw=0.8,
                    capsize=2)
    ax.bar(0, 0, 0, color=color, label=label)


def load_v(m, name='uap_xi10.pt'):
    v = torch.load(RESULTS_DIR / m / name, map_location='cpu').float()
    return v.reshape(3, *v.shape[-2:])


def v_image(v):
    """Perturbation as an image, contrast stretched: gray = 0, saturated = ±max."""
    return (v / (2 * v.abs().max()) + 0.5).permute(1, 2, 0).numpy()
