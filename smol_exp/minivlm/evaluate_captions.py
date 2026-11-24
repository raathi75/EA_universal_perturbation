"""Captions of the trained mini-VLM on the held-out ImageNetV2 images (same images as the SmolVLM notebooks),
clean and perturbed, and share of captions changed (MiniLM similarity < 0.5, and < 0.7 as a stricter check),
with 95% Wilson intervals, on the first 200 images (as in the report) and on all --n_caption images.

Perturbations: this encoder's universal v, random noise of the same budget, and, when they exist, the universal
v of the other encoder and of SmolVLM-256M (transfer). Perturbed images are rounded to 8 bits.
Can be relaunched: perturbations already evaluated are skipped (e.g. relaunch once the other machine is done,
to add the transfer from the other encoder).
"""
import argparse
import json

import torch

from common import (extra_conditions, MODELS, CAPTION_SIM_CHANGED, ENCODERS, HERE, IMG_SIZE, SEED, TextSim, get_device, imagenetv2_images,
                    load_minivlm, load_perturbation, other_encoders, quantize, results_dir, wilson)

N_REPORT = 200   # number of captioned images in the SmolVLM report


def rates(sim, threshold):
    out = {}
    for name, n in [('first%d' % N_REPORT, min(N_REPORT, len(sim))), ('all', len(sim))]:
        k = int((sim[:n] < threshold).sum())
        lo, hi = wilson(k, n)
        out[name] = {'n': n, 'changed': k, 'rate': 100 * k / n, 'ci95': [lo, hi]}
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Caption change of the mini-VLM under universal perturbations')
    parser.add_argument('--encoder', required=True, choices=MODELS)
    parser.add_argument('--xi', type=float, default=10)
    parser.add_argument('--n_caption', type=int, default=500, help='held-out images captioned (the report: 200)')
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    out = results_dir(args.encoder)
    file_res = out / ('eval_captions_stage%d_xi%g.json' % (args.stage, args.xi))
    file_caps = out / ('captions_stage%d_xi%g.json' % (args.stage, args.xi))
    res = json.load(open(file_res)) if file_res.exists() else {'encoder': args.encoder, 'xi': args.xi, 'results': {}}
    caps = json.load(open(file_caps)) if file_caps.exists() else {}

    vlm = load_minivlm(args.encoder, device, stage=args.stage)
    sim_fn = TextSim(device)
    _, eval_u8, _, eval_files = imagenetv2_images()
    x = eval_u8[:args.n_caption].to(device).float() / 255
    res['files'] = eval_files[:args.n_caption]

    if 'clean' not in caps or len(caps['clean']) != len(x):
        print('>> Clean captions of %d images' % len(x))
        caps['clean'] = vlm.caption(x, args.batch_size)
        caps.pop('clean_repeat', None)
    if 'clean_repeat' not in caps:
        # Sanity check: the first batch decoded again (same batch size, since bf16 results depend slightly on the
        # batch shape) should give the same greedy captions
        caps.pop('clean_again', None)
        caps['clean_repeat'] = vlm.caption(x[:args.batch_size], args.batch_size)
        json.dump(caps, open(file_caps, 'w'), indent=1)
    same = sum(a == b for a, b in zip(caps['clean_repeat'], caps['clean']))
    res['clean_repeat_identical'] = [same, len(caps['clean_repeat'])]
    print('>> Determinism check: %d/%d identical captions when decoding again' % (same, len(caps['clean_repeat'])))
    if same < len(caps['clean_repeat']):
        print('!! Warning: decoding is not fully reproducible (GPU numerics); results are still usable')

    tag = 'uap_xi%g.pt' % args.xi
    torch.manual_seed(SEED)
    perts = {'universal v (%s)' % args.encoder: out / tag,
             'universal v (%s, connector-output)' % args.encoder: out / ('uap_conn_xi%g.pt' % args.xi),
             # end-to-end, likelihood of the clean captions (uap_encoder.py --target caption)
             'universal v (%s, caption loss)' % args.encoder: out / ('uap_cap_xi%g.pt' % args.xi),
             'random ±xi': args.xi / 255 * torch.sign(torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)),
             **{'universal v (%s), transfer' % o: results_dir(o) / tag for o in other_encoders(args.encoder)},
             'universal v (SmolVLM-256M, 512 -> 224), transfer': HERE.parent / 'uap_smolvlm' / ('uap_smolvlm256m_linf%g.npy' % args.xi)}
    perts.update(extra_conditions(out, args.xi, IMG_SIZE, device, 'universal v (%s)' % args.encoder))

    for name, p in perts.items():
        if name in res['results']:
            continue
        if not torch.is_tensor(p):
            if not p.exists():
                print('>> Skipped (not computed yet):', name, p)
                continue
            p = load_perturbation(p, device)
        print('>> Captions with', name)
        caps[name] = vlm.caption(quantize(x + p), args.batch_size)
        sim = sim_fn(caps['clean'], caps[name])
        res['results'][name] = {'mean_caption_sim': sim.mean().item(),
                                'changed@0.5': rates(sim, CAPTION_SIM_CHANGED), 'changed@0.7': rates(sim, 0.7)}
        json.dump(caps, open(file_caps, 'w'), indent=1)
        json.dump(res, open(file_res, 'w'), indent=1)

    print('\n%s, stage %d, xi = %g/255: captions changed (95%% CI)' % (args.encoder, args.stage, args.xi))
    print('%-50s %22s %22s %10s' % ('perturbation', 'first 200 @0.5', 'all @0.5', 'all @0.7'))
    for name, r in res['results'].items():
        a, b, c = r['changed@0.5']['first%d' % N_REPORT], r['changed@0.5']['all'], r['changed@0.7']['all']
        print('%-50s %6.1f%% (%4.1f-%4.1f) %6.1f%% (%4.1f-%4.1f) %9.1f%%' % (
            name, a['rate'], *a['ci95'], b['rate'], *b['ci95'], c['rate']))
    print('\nExamples (clean -> universal v):')
    key = 'universal v (%s)' % args.encoder
    for c0, c1 in list(zip(caps['clean'], caps.get(key, [])))[:5]:
        print('  ', c0, '\n   ->', c1)
    print('>> Saved', file_res, 'and', file_caps)
