"""Systematic search of the words injected by each perturbation (no hand-chosen regex).
A word is "injected" by a condition when it is in <= 1 % of the clean captions and in >= 10 % of the perturbed captions
of the same model and the same images (a word counts once per caption; stop words removed). Own perturbations only
(transfers and random noise excluded). Reads captions_stage2_xi10.json (ImageNetV2 held-out) and
coco_captions_stage2.json (COCO val) -> results/injection_scan.json, and prints a table.
  python injection_scan.py
"""
import json
import re
from collections import Counter

from common import RESULTS_DIR

MODELS = ['siglip', 'clip', 'ijepa', 'mae', 'smolvlm-256m', 'llava-ov-0.5b', 'qwen2-vl-2b', 'qwen2-vl-2b-224']
STOP = set('a an the of and with in on is are to its it this that image features scene background set by or'.split())
CLEAN_MAX, PERT_MIN = 0.01, 0.10


def words(caps):
    return Counter(w for s in caps for w in set(re.findall(r'[a-z]+', s.lower())))


def scan(caps):
    n = len(caps['clean'])
    wc = words(caps['clean'])
    out = {}
    for k, pert in caps.items():
        if k.startswith('clean') or 'transfer' in k or 'random' in k:
            continue
        wk = words(pert)
        g = sorted(((w, c) for w, c in wk.items() if w not in STOP and wc[w] <= CLEAN_MAX * n and c >= PERT_MIN * n),
                   key=lambda x: -x[1])
        # share of captions that contain at least one injected word
        inj = set(w for w, _ in g)
        any_rate = sum(bool(inj & set(re.findall(r'[a-z]+', s.lower()))) for s in pert) / n if inj else 0.0
        out[k] = {'words': {w: c / n for w, c in g}, 'any_injected_word': any_rate, 'n': n}
    return out


if __name__ == '__main__':
    res = {}
    for m in MODELS:
        res[m] = {}
        for f, ds in [('captions_stage2_xi10.json', 'imagenetv2'), ('coco_captions_stage2.json', 'coco')]:
            p = RESULTS_DIR / m / f
            if p.exists():
                res[m][ds] = scan(json.load(open(p)))
    json.dump(res, open(RESULTS_DIR / 'injection_scan.json', 'w'), indent=1)
    print('%-16s %-48s %-9s %-9s %s' % ('model', 'condition', 'INv2 any', 'COCO any', 'injected words (ImageNetV2)'))
    for m in MODELS:
        for k, r in res[m].get('imagenetv2', {}).items():
            rc = res[m].get('coco', {}).get(k, {'any_injected_word': float('nan')})
            print('%-16s %-48s %7.1f%% %7.1f%%  %s' % (m, k[:48], 100 * r['any_injected_word'], 100 * rc['any_injected_word'],
                                                    ', '.join('%s %.0f%%' % (w, 100 * x) for w, x in list(r['words'].items())[:6])))
    print('\n>> Saved', RESULTS_DIR / 'injection_scan.json')
