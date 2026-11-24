"""Attack evaluated on COCO val2017 against the HUMAN captions (5 per image), not only against the model's own
clean caption. COCO val2017 is never used for training (the connector is trained on train2017).

For the clean images and for each perturbation (own universal v, random ±ξ, v of the other encoders and of
SmolVLM-256M; all built on ImageNetV2, so this also tests their universality across datasets):
  - CIDEr against the 5 human captions (corpus level): caption quality, the clean row is the quality check
    that the mini-VLMs must pass before being compared;
  - wrong: the caption is far from every human caption (best MiniLM similarity < 0.5), with a 95% interval.
    This measures whether captions become WRONG, not merely different;
  - changed: MiniLM similarity with the clean caption < 0.5 (as in the report), with a 95% interval.
Perturbed images are rounded to 8 bits. Can be relaunched: conditions already evaluated are skipped.
"""
import argparse
import json
import re

import torch

from common import (extra_conditions, MODELS, CAPTION_SIM_CHANGED, ENCODERS, HERE, IMG_SIZE, SEED, CocoCaptions, TextSim, get_device,
                    load_minivlm, load_perturbation, other_encoders, quantize, results_dir, wilson)


def tokenize(s):
    return ' '.join(re.findall(r"[a-z0-9]+", s.lower()))


def rate(flags):
    k, n = int(flags.sum()), len(flags)
    lo, hi = wilson(k, n)
    return {'n': n, 'count': k, 'rate': 100 * k / n, 'ci95': [lo, hi]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mini-VLM on COCO val2017: clean quality and attack vs human captions')
    parser.add_argument('--encoder', required=True, choices=MODELS)
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2])
    parser.add_argument('--xi', type=float, default=10)
    parser.add_argument('--n_images', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    from pycocoevalcap.cider.cider import Cider

    device = get_device()
    out = results_dir(args.encoder)
    file_res = out / ('coco_stage%d.json' % args.stage)
    file_caps = out / ('coco_captions_stage%d.json' % args.stage)
    res = json.load(open(file_res)) if file_res.exists() else {'encoder': args.encoder, 'stage': args.stage, 'results': {}}
    caps = json.load(open(file_caps)) if file_caps.exists() else {}

    vlm = load_minivlm(args.encoder, device, stage=args.stage)
    sim_fn = TextSim(device)
    data = CocoCaptions('val2017', max_images=args.n_images)
    x = torch.stack([data[i][0] for i in range(len(data))]).to(device).float() / 255
    refs = [r for _, r in data.items]
    gts = {i: [tokenize(c) for c in r] for i, r in enumerate(refs)}
    res['n_images'] = len(data)
    res['files'] = [f.split('/')[-1] for f, _ in data.items]

    # Human captions embedded once: best similarity of a model caption to any of the 5 human captions
    flat = [c for r in refs for c in r]
    ref_emb = sim_fn.st.encode(flat, convert_to_tensor=True, normalize_embeddings=True)
    owner = torch.tensor([i for i, r in enumerate(refs) for _ in r], device=ref_emb.device)

    def best_ref_sim(captions):
        e = sim_fn.st.encode(captions, convert_to_tensor=True, normalize_embeddings=True)
        s = (e[owner] * ref_emb).sum(-1)
        best = torch.full((len(captions),), -1.0, device=s.device).scatter_reduce(0, owner, s, reduce='amax')
        return best.cpu()

    torch.manual_seed(SEED)
    tag = 'uap_xi%g.pt' % args.xi
    conds = {'clean': None,
             'universal v (%s)' % args.encoder: out / tag,
             # built on the connector outputs (what the language model receives), uap_encoder.py --target connector
             'universal v (%s, connector-output)' % args.encoder: out / ('uap_conn_xi%g.pt' % args.xi),
             # end-to-end, likelihood of the clean captions (uap_encoder.py --target caption)
             'universal v (%s, caption loss)' % args.encoder: out / ('uap_cap_xi%g.pt' % args.xi),
             'random ±xi': args.xi / 255 * torch.sign(torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)),
             **{'universal v (%s), transfer' % o: results_dir(o) / tag for o in other_encoders(args.encoder)},
             'universal v (SmolVLM-256M, 512 -> 224), transfer':
                 HERE.parent / 'uap_smolvlm' / ('uap_smolvlm256m_linf%g.npy' % args.xi)}
    conds.update(extra_conditions(out, args.xi, IMG_SIZE, device, 'universal v (%s)' % args.encoder))

    for name, p in conds.items():
        if name in res['results']:
            continue
        if p is not None and not torch.is_tensor(p):
            if not p.exists():
                print('>> Skipped (not computed yet):', name, p)
                continue
            p = load_perturbation(p, device)
        print('>> Captions of %d COCO val images: %s' % (len(x), name))
        caps[name] = vlm.caption(x if p is None else quantize(x + p), args.batch_size)
        cider, _ = Cider().compute_score(gts, {i: [tokenize(c)] for i, c in enumerate(caps[name])})
        best = best_ref_sim(caps[name])
        r = {'CIDEr': cider, 'mean_best_human_sim': best.mean().item(), 'wrong': rate(best < CAPTION_SIM_CHANGED)}
        if p is not None:
            r['changed'] = rate(sim_fn(caps['clean'], caps[name]) < CAPTION_SIM_CHANGED)
        res['results'][name] = r
        json.dump(caps, open(file_caps, 'w'), indent=1)
        json.dump(res, open(file_res, 'w'), indent=1)

    print('\n%s, stage %d, %d COCO val images, xi = %g/255' % (args.encoder, args.stage, len(x), args.xi))
    print('%-50s %7s %22s %22s' % ('condition', 'CIDEr', 'wrong (far from humans)', 'changed (vs clean)'))
    for name, r in res['results'].items():
        w = r['wrong']
        c = r.get('changed')
        print('%-50s %7.3f %6.1f%% (%4.1f-%4.1f)  %s' % (
            name, r['CIDEr'], w['rate'], *w['ci95'],
            '%6.1f%% (%4.1f-%4.1f)' % (c['rate'], *c['ci95']) if c else '      —'))
    print('\nExamples (human / clean / universal v):')
    key = 'universal v (%s)' % args.encoder
    for i in range(min(4, len(refs))):
        print('  human:', refs[i][0], '\n  clean:', caps['clean'][i], '\n  v    :', caps.get(key, ['—'] * len(refs))[i], '\n')
    print('>> Saved', file_res, 'and', file_caps)
