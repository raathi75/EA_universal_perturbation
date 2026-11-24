"""ImageNet accuracy read from the captions (no extra model run): each caption of the 500 ImageNetV2 held-out images
is matched (MiniLM, the similarity of the report) against the 1000 class names "a photo of a <name>"; top-1 / top-5
accuracy against the true class (folder of the image). Captions are short and generic, so absolute numbers are low
for every model: compare models and conditions, not with a classifier.

Uses the caption files already saved: captions_stage2_xi10.json (clean, own v, random, transfers) and
defense_ft_captions.json (clean / fixed v / adaptive v of no-defense, FT-fixed, FT-UAT).
Writes results/imagenet_accuracy.json and prints the tables.
"""
import json
from pathlib import Path

import torch

from common import RESULTS_DIR

SYNSETS = Path(__file__).resolve().parent / 'imagenet_classes.txt'   # 'n01440764 tench, Tinca tinca', class order of ImageNetV2


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return 100 * (c - h), 100 * (c + h)


if __name__ == '__main__':
    from sentence_transformers import SentenceTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    names = [l.strip().split(' ', 1)[1] for l in open(SYNSETS)]
    # every synonym of a class is a prompt; a class scores its best synonym
    prompts, owner = [], []
    for c, n in enumerate(names):
        for s in n.split(', '):
            prompts.append('a photo of a %s' % s)
            owner.append(c)
    owner = torch.tensor(owner, device=device)
    p_emb = st.encode(prompts, convert_to_tensor=True, normalize_embeddings=True, batch_size=256)

    # held-out files, in the order of the caption files (saved by uap_encoder.py; same split for every model)
    eval_files = json.load(open(RESULTS_DIR / 'siglip' / 'uap_xi10_tokens.json'))['eval_files']
    labels = torch.tensor([int(Path(f).parent.name) for f in eval_files], device=device)

    def accuracy(caps):
        e = st.encode(caps, convert_to_tensor=True, normalize_embeddings=True, batch_size=256)
        s = e @ p_emb.T
        cls = torch.full((len(caps), 1000), -2.0, device=device).scatter_reduce(
            1, owner.expand(len(caps), -1), s, reduce='amax')
        top5 = cls.topk(5, dim=1).indices
        t1 = (top5[:, 0] == labels).sum().item()
        t5 = (top5 == labels[:, None]).any(1).sum().item()
        n = len(caps)
        return {'top1': 100 * t1 / n, 'top1_ci95': wilson(t1, n), 'top5': 100 * t5 / n, 'top5_ci95': wilson(t5, n), 'n': n}

    out = {}
    for d in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name != 'old_10epochs'):
        m, r = d.name, {}
        f = d / 'captions_stage2_xi10.json'
        if f.exists():
            for k, caps in json.load(open(f)).items():
                if k != 'clean_repeat' and len(caps) == len(labels):
                    r[k] = accuracy(caps)
        f = d / 'defense_ft_captions.json'
        if f.exists():
            for k, caps in json.load(open(f)).items():
                if k.startswith('imagenetv2') and len(caps) == len(labels):
                    r['defense | ' + k.split(': ', 1)[1]] = accuracy(caps)
        if r:
            out[m] = r
    json.dump(out, open(RESULTS_DIR / 'imagenet_accuracy.json', 'w'), indent=1)

    for m, r in out.items():
        print('\n== %s: ImageNetV2 held-out (500), accuracy read from the captions, top-1 | top-5' % m)
        for k, a in r.items():
            if 'transfer' in k:
                continue
            print('  %-50s %5.1f%% | %5.1f%%' % (k, a['top1'], a['top5']))
    print('\n>> Saved', RESULTS_DIR / 'imagenet_accuracy.json')
