"""Sanity check of the attack on a real VLM: can ONE image be fooled with a small perturbation of its own?
(per-image attack, the easiest case for the attacker; a universal v must fool all images with the same pattern)

For each of the first N held-out ImageNetV2 images: PGD (sign of the gradient, l_inf ball xi, rounded to 8 bits at
the end) that lowers the likelihood of the model's own clean caption, through the whole model (same end-to-end loss
as real_vlm.py --target caption). Several budgets xi. Reports the share of captions changed (MiniLM similarity
< 0.5 with the clean caption, as everywhere) and examples.
If even this fails, the attack pipeline is at fault; if it succeeds easily, the model's robustness is specific to
UNIVERSAL perturbations.

  python single_image.py --model qwen2-vl-2b            (results/<model>/single_image.json)
"""
import argparse
import json

import torch

from common import CAPTION_SIM_CHANGED, SEED, TextSim, get_device, results_dir
from real_vlm import SPECS, RealVLM, imagenetv2_at

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-image end-to-end attack on a real VLM (sanity check)')
    parser.add_argument('--model', required=True, choices=list(SPECS))
    parser.add_argument('--n', type=int, default=40, help='number of held-out images')
    parser.add_argument('--xis', default='2,4,10', help='budgets on the [0, 255] scale')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    out = results_dir(args.model)
    vlm = RealVLM(args.model, device)
    S = vlm.size
    _, eval_u8 = imagenetv2_at(S)
    imgs = eval_u8[:args.n]
    vlm.sanity_check(imgs)
    vlm.setup_caption_attack(imgs, out / ('single_image_caption_ids_%d.json' % args.n))
    clean_caps = [vlm.processor.tokenizer.decode(c, skip_special_tokens=True).strip() for c in vlm.cap_ids]
    sim_fn = TextSim(device)
    f_res = out / 'single_image.json'
    res = json.load(open(f_res)) if f_res.exists() else {'model': args.model, 'n': args.n, 'steps': args.steps,
                                                         'clean': clean_caps, 'results': {}}

    for xi_255 in [float(x) for x in args.xis.split(',')]:
        key = 'xi %g/255' % xi_255
        if key in res['results']:
            continue
        xi, alpha = xi_255 / 255, 2.5 * xi_255 / 255 / args.steps   # step of Madry et al.: 2.5 xi / steps
        torch.manual_seed(SEED)
        adv_u8, scores = [], []
        for i in range(0, len(imgs), args.batch_size):
            x = imgs[i:i + args.batch_size].to(device).float() / 255
            idx = torch.arange(i, i + len(x))
            delta = torch.zeros_like(x, requires_grad=True)
            for step in range(args.steps):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    score = vlm.caption_score((x + delta).clamp(0, 1), idx)
                g, = torch.autograd.grad(score.sum(), [delta])
                with torch.no_grad():
                    delta -= alpha * g.sign()
                    delta.clamp_(-xi, xi)
            with torch.no_grad():
                u8 = torch.round((x + delta).clamp(0, 1) * 255).to(torch.uint8)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    scores += vlm.caption_score(u8.float() / 255, idx).float().cpu().tolist()
            adv_u8.append(u8.cpu())
            print('  %s: images %d-%d done, caption likelihood %.3f' % (key, i, i + len(x) - 1, sum(scores[-len(x):]) / len(x)))
        adv_caps = vlm.caption(torch.cat(adv_u8), None, args.batch_size)
        sim = sim_fn(clean_caps, adv_caps)
        res['results'][key] = {'changed@0.5': {'rate': 100 * (sim < CAPTION_SIM_CHANGED).float().mean().item()},
                               'mean_caption_sim': sim.mean().item(), 'caption_likelihood': sum(scores) / len(scores),
                               'captions': adv_caps}
        json.dump(res, open(f_res, 'w'), indent=1)
        print('>> %s, %s: %.0f%% of the %d captions changed (mean similarity %.2f)' % (
            args.model, key, res['results'][key]['changed@0.5']['rate'], len(imgs), sim.mean().item()))

    print('\n%s, per-image attack (%d images, %d PGD steps), captions changed:' % (args.model, args.n, args.steps))
    for key, r in res['results'].items():
        print('  %-12s %5.0f%%   (caption likelihood %.3f)' % (key, r['changed@0.5']['rate'], r['caption_likelihood']))
    print('\nExamples (clean -> attacked at the largest budget):')
    last = list(res['results'])[-1]
    for k in range(min(4, len(res['clean']))):
        print('  %s\n  -> %s' % (res['clean'][k], res['results'][last]['captions'][k]))
    print('>> Saved', f_res)
