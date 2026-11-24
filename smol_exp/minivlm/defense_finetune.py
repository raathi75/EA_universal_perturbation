"""Defense by fine-tuning the vision encoder, with the recipe of the SmolVLM report (notebook 07, part B),
applied to each mini-VLM. The connector and the language model are NOT retrained (as in the report, where
SmolVLM's language model was untouched).

Two variants, both on the 500 ImageNetV2 build images, 3 epochs, Adam lr 1e-5, batch 8:
  FT-fixed: the encoder learns to resist the FIXED universal v of uap_encoder.py (what the original UAP paper did)
  FT-UAT:   universal adversarial training (Shafahi et al. 2020): after each encoder step, one Adam step on v
            (lr xi/10) against the current encoder, so the encoder faces an evolving attacker
Loss (feature anchoring): the tokens of the perturbed AND of the clean image must stay close to the tokens of the
ORIGINAL encoder on the clean image (1 - cosine), which keeps the encoder compatible with the trained connector.
The original tokens are computed once, so no second copy of the encoder is kept on the GPU.

Then, for each fine-tuned encoder:
  - adaptive v: a new universal perturbation against the fine-tuned encoder (warm start from the original v,
    Adam lr xi/10, as in the report), run to convergence (at least the 7 epochs of the report);
  - 'no-defense': the same adaptive attack against the ORIGINAL encoder, the reference at equal attack effort;
  - token damage on the held-out ImageNetV2 images, and clean drift (token cosine with the original encoder);
  - captions on COCO val (CIDEr and wrong vs the human captions, changed) and on the ImageNetV2 held-out images
    (changed, as in the report): clean cost, fixed v, adaptive v.

Fine-tuned encoders go to the local disk (MINIVLM_DATA/defense, ~2.5 GB each for a ViT-H), results to
results/<encoder>/defense_ft.json. Can be relaunched: finished steps are skipped.
If the GPU runs out of memory: --train_blocks 8 only fine-tunes the last 8 transformer blocks.
"""
import argparse
import json
import re
import time

import numpy as np
import torch

from common import (CAPTION_SIM_CHANGED, DATA_DIR, ENCODERS, IMG_SIZE, SEED, XI, CocoCaptions, Encoder, TextSim,
                    amp_dtype, converge_uap, get_device, imagenetv2_images, load_minivlm, load_perturbation, quantize, results_dir,
                    token_cosine)
from eval_coco import rate, tokenize
from evaluate_captions import rates

# 'no-defense': the original encoder, attacked by the same adaptive procedure (warm start from the original v,
# 7 more epochs): the reference at EQUAL attack effort, since the adaptive v gets more epochs than the original v
VARIANTS = ['no-defense', 'FT-fixed', 'FT-UAT']


def to01(u8, device):
    return u8.to(device).float() / 255


def autocast(device):
    bf16 = device.type == 'cuda' and amp_dtype() == torch.bfloat16
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16)


@torch.no_grad()
def tokens_of(encoder, images_u8, device, v=None, bs=32):
    out = []
    for i in range(0, len(images_u8), bs):
        x = to01(images_u8[i:i + bs], device)
        with autocast(device):
            out.append(encoder(x if v is None else quantize(x + v)).to(torch.bfloat16))
    return torch.cat(out)


def set_trainable(encoder, n_blocks):
    """All parameters (n_blocks < 0, as in the report) or only the last n_blocks transformer blocks."""
    names = [k for k, _ in encoder.net.named_parameters()]
    layer = {k: int(m.group(1)) for k in names for m in [re.search(r'(?:^|\.)layers?\.(\d+)\.', k)] if m}
    n_layers = max(layer.values()) + 1
    params = []
    for k, p in encoder.net.named_parameters():
        train = n_blocks < 0 or (layer[k] >= n_layers - n_blocks if k in layer else 'embeddings' not in k)
        p.requires_grad_(train)
        if train:
            params.append(p)
    print('>> Fine-tuning %.1fM parameters (%s)' % (
        sum(p.numel() for p in params) / 1e6, 'all' if n_blocks < 0 else 'last %d of %d blocks' % (n_blocks, n_layers)))
    return params


def finetune(encoder, variant, build_u8, z_ref, v_fixed, args, device):
    params = set_trainable(encoder, args.train_blocks)
    encoder.train()
    try:
        # Non-reentrant: gradients reach the trainable blocks even when their input does not require grad
        encoder.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    except Exception as e:  # noqa: BLE001
        print('>> No gradient checkpointing:', e)
    opt = torch.optim.Adam(params, lr=args.lr, foreach=False)   # foreach=False: no extra copy of the weights
    v = v_fixed.clone()
    if variant == 'FT-UAT':
        v.requires_grad_(True)
        opt_v = torch.optim.Adam([v], lr=XI / 10)
    g = torch.Generator().manual_seed(SEED)
    hist = []
    for ep in range(args.epochs):
        t0, la, lc, lv = time.time(), [], [], []
        for idx in torch.randperm(len(build_u8), generator=g).split(args.batch_size):
            x, zr = to01(build_u8[idx], device), z_ref[idx]
            # Encoder step: perturbed and clean tokens close to the ORIGINAL clean tokens (two backward passes
            # to keep the GPU memory low, the gradients add up)
            opt.zero_grad(set_to_none=True)
            with autocast(device):
                l_adv = (1 - token_cosine(encoder((x + v.detach()).clamp(0, 1)), zr)).mean()
            l_adv.backward()
            with autocast(device):
                l_cln = (1 - token_cosine(encoder(x), zr)).mean()
            (args.anchor_w * l_cln).backward()
            opt.step()
            la.append(l_adv.item())
            lc.append(l_cln.item())
            if variant == 'FT-UAT':
                # Attacker step against the current encoder (gradient with respect to v only)
                with torch.no_grad(), autocast(device):
                    z0 = encoder(x)
                with autocast(device):
                    loss_v = token_cosine(z0, encoder((x + v).clamp(0, 1))).mean()
                v.grad = torch.autograd.grad(loss_v, [v])[0]
                opt_v.step()
                with torch.no_grad():
                    v.clamp_(-XI, XI)
                lv.append(loss_v.item())
        h = {'epoch': ep + 1, 'perturbed_vs_original': float(np.mean(la)), 'clean_drift': float(np.mean(lc)),
             'minutes': (time.time() - t0) / 60}
        if lv:
            h['attacker_token_cos'] = float(np.mean(lv))
        hist.append(h)
        print('%s %s epoch %d/%d: perturbed-vs-original loss %.3f | clean drift loss %.4f%s | %.1f min' % (
            args.encoder, variant, ep + 1, args.epochs, h['perturbed_vs_original'], h['clean_drift'],
            ' | attacker token cos %.3f' % h['attacker_token_cos'] if lv else '', h['minutes']))
    try:
        encoder.net.gradient_checkpointing_disable()
    except Exception:  # noqa: BLE001
        pass
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return hist


def adaptive_uap(encoder, build_u8, v_fixed, args, device, ckpt_path):
    """New universal v against the (fine-tuned) encoder, warm start from the original v as in the report (Adam,
    lr xi/10), run to CONVERGENCE with common.converge_uap: best v on the build images kept (never the held-out
    ones), step halved on plateaus, at least the 7 epochs of the report."""
    v, st = converge_uap(encoder, build_u8, device, v0=v_fixed, xi=XI, lr=XI / 10, batch_size=8, ckpt_path=ckpt_path,
                         min_epochs=args.adapt_epochs, max_epochs=args.adapt_max_epochs, name='adaptive v')
    return v, st, st['best_epoch']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Defense: fine-tune the vision encoder of a mini-VLM (report, notebook 07)')
    parser.add_argument('--encoder', required=True, choices=list(ENCODERS))
    parser.add_argument('--variants', nargs='+', default=VARIANTS, choices=VARIANTS)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--anchor_w', type=float, default=1.0)
    parser.add_argument('--adapt_epochs', type=int, default=7, help='minimum epochs of the adaptive attack (report: 7)')
    parser.add_argument('--adapt_max_epochs', type=int, default=60)
    parser.add_argument('--train_blocks', type=int, default=-1, help='-1: whole encoder (report); N: last N blocks')
    parser.add_argument('--n_coco', type=int, default=1000)
    parser.add_argument('--caption_batch_size', type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    out = results_dir(args.encoder)
    tag = 'xi%g' % (XI * 255)
    file_res, file_caps = out / 'defense_ft.json', out / 'defense_ft_captions.json'
    res = json.load(open(file_res)) if file_res.exists() else {'encoder': args.encoder, 'config': vars(args), 'variants': {}}
    caps = json.load(open(file_caps)) if file_caps.exists() else {}
    ft_dir = DATA_DIR / 'defense'
    ft_dir.mkdir(parents=True, exist_ok=True)

    def save():
        json.dump(res, open(file_res, 'w'), indent=1)
        json.dump(caps, open(file_caps, 'w'), indent=1)

    torch.manual_seed(SEED)
    build_u8, eval_u8, _, _ = imagenetv2_images()
    v_fixed = load_perturbation(out / ('uap_%s.pt' % tag), device)

    # ------------------------------------------------------------------ 1. fine-tuning and adaptive v (no LM)
    need_ft = [var for var in args.variants if var != 'no-defense'
               if not (ft_dir / ('%s_%s.pt' % (args.encoder, var))).exists()
               and 'captions' not in res['variants'].get(var, {})]
    # (an adaptive v without 'adaptive_best_epoch' was made by the first version, which kept the last epoch: redone)
    need_adapt = [var for var in args.variants if not (out / ('adaptive_%s_%s.pt' % (tag, var))).exists()
                  or 'adaptive_best_epoch' not in res['variants'].get(var, {})]
    if need_ft or need_adapt:
        encoder = Encoder(args.encoder).to(device)
        original = {k: t.detach().cpu().clone() for k, t in encoder.state_dict().items()}
        print('>> Original tokens of the %d build images (anchor)' % len(build_u8))
        z_ref = tokens_of(encoder, build_u8, device)
        z_eval = tokens_of(encoder, eval_u8, device)
        for var in args.variants:
            r = res['variants'].setdefault(var, {})
            path_ft = ft_dir / ('%s_%s.pt' % (args.encoder, var))
            path_v = out / ('adaptive_%s_%s.pt' % (tag, var))
            if var not in need_ft and var not in need_adapt:
                continue
            encoder.load_state_dict(original)
            if var == 'no-defense':
                print('>> Original encoder (no defense)')
            elif path_ft.exists():
                encoder.load_state_dict(torch.load(path_ft, map_location='cpu'))
                print('>> Loaded', path_ft)
            elif var not in need_ft:
                print('>> Skipped %s: fine-tuned encoder not on this machine (%s)' % (var, path_ft))
                continue
            else:
                print('=== Fine-tuning %s, %s ===' % (args.encoder, var))
                r['train_history'] = finetune(encoder, var, build_u8, z_ref, v_fixed, args, device)
                torch.save(encoder.state_dict(), path_ft)
                print('>> Saved', path_ft)
            # Clean drift and token damage of the fixed v on the held-out images
            z_ft = tokens_of(encoder, eval_u8, device)
            r['clean_drift_token_cos'] = token_cosine(z_eval, z_ft).mean().item()
            tc = r.setdefault('token_cos_heldout', {})
            tc['fixed v'] = token_cosine(z_ft, tokens_of(encoder, eval_u8, device, v_fixed)).mean().item()
            if var in need_adapt:
                print('=== Adaptive v against %s, %s ===' % (args.encoder, var))
                torch.manual_seed(SEED)
                ck = out / ('adaptive_%s_%s_ckpt.pt' % (tag, var))
                v_ad, r['adaptive_history'], r['adaptive_best_epoch'] = adaptive_uap(encoder, build_u8, v_fixed, args, device, ck)
                torch.save(v_ad, path_v)
                ck.unlink(missing_ok=True)
                for k in ['captions', 'coco', 'imagenetv2']:   # captions to be redone with this v
                    r.pop(k, None)
            tc['adaptive v'] = token_cosine(z_ft, tokens_of(encoder, eval_u8, device, load_perturbation(path_v, device))).mean().item()
            print('%s %s: clean drift token cos %.4f | token cos fixed v %.3f, adaptive v %.3f (held-out)' % (
                args.encoder, var, r['clean_drift_token_cos'], tc['fixed v'], tc['adaptive v']))
            save()
        del encoder, z_ref, z_eval
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 2. captions with each fine-tuned encoder
    from pycocoevalcap.cider.cider import Cider

    vlm = load_minivlm(args.encoder, device, stage=2)
    original = {k: t.detach().cpu().clone() for k, t in vlm.encoder.state_dict().items()}
    sim_fn = TextSim(device)
    coco = CocoCaptions('val2017', max_images=args.n_coco)
    x_coco = torch.stack([coco[i][0] for i in range(len(coco))]).to(device).float() / 255
    refs = [r for _, r in coco.items]
    gts = {i: [tokenize(c) for c in r] for i, r in enumerate(refs)}
    ref_emb = sim_fn.st.encode([c for r in refs for c in r], convert_to_tensor=True, normalize_embeddings=True)
    owner = torch.tensor([i for i, r in enumerate(refs) for _ in r], device=ref_emb.device)
    x_inv2 = to01(eval_u8, device)

    def coco_metrics(c, clean=None):
        cider, _ = Cider().compute_score(gts, {i: [tokenize(s)] for i, s in enumerate(c)})
        e = sim_fn.st.encode(c, convert_to_tensor=True, normalize_embeddings=True)
        s = (e[owner] * ref_emb).sum(-1)
        best = torch.full((len(c),), -1.0, device=s.device).scatter_reduce(0, owner, s, reduce='amax').cpu()
        m = {'CIDEr': cider, 'mean_best_human_sim': best.mean().item(), 'wrong': rate(best < CAPTION_SIM_CHANGED)}
        if clean is not None:
            m['changed'] = rate(sim_fn(clean, c) < CAPTION_SIM_CHANGED)
        return m

    def changed(clean, c):
        sim = sim_fn(clean, c)
        return {'mean_caption_sim': sim.mean().item(), 'changed@0.5': rates(sim, CAPTION_SIM_CHANGED),
                'changed@0.7': rates(sim, 0.7)}

    # Clean captions of the ORIGINAL mini-VLM (reused from eval_coco.py / evaluate_captions.py when available)
    def original_clean(key, x, path, n):
        if key in caps:
            return caps[key]
        old = json.load(open(path)) if path.exists() else {}
        if len(old.get('clean', [])) == n:
            caps[key] = old['clean']
        else:
            print('>> Clean captions of the original mini-VLM:', key)
            vlm.encoder.load_state_dict(original)
            caps[key] = vlm.caption(x, args.caption_batch_size)
        return caps[key]

    clean0_coco = original_clean('coco: clean, original encoder', x_coco, out / 'coco_captions_stage2.json', len(x_coco))
    clean0_inv2 = original_clean('imagenetv2: clean, original encoder', x_inv2, out / ('captions_stage2_%s.json' % tag), len(x_inv2))
    save()

    for var in args.variants:
        r = res['variants'].setdefault(var, {})
        if 'captions' in r:
            continue
        path_ft = ft_dir / ('%s_%s.pt' % (args.encoder, var))
        if var == 'no-defense':
            vlm.encoder.load_state_dict(original)
        elif not path_ft.exists():
            print('>> Skipped %s: fine-tuned encoder not found (%s). Run on the machine that trained it.' % (var, path_ft))
            continue
        else:
            vlm.encoder.load_state_dict(torch.load(path_ft, map_location='cpu'))
        v_ad = load_perturbation(out / ('adaptive_%s_%s.pt' % (tag, var)), device)
        r['coco'], r['imagenetv2'] = {}, {}
        print('=== Captions with the %s encoder (%d COCO val, %d ImageNetV2 held-out images) ===' % (
            var, len(x_coco), len(x_inv2)))
        for name, v in [('clean', None), ('fixed v', v_fixed), ('adaptive v', v_ad)]:
            k = '%s: %s' % (var, name)
            caps['coco: ' + k] = vlm.caption(x_coco if v is None else quantize(x_coco + v), args.caption_batch_size)
            caps['imagenetv2: ' + k] = vlm.caption(x_inv2 if v is None else quantize(x_inv2 + v), args.caption_batch_size)
            if v is None:   # cost of the defense: clean captions vs those of the original mini-VLM
                r['coco'][name] = coco_metrics(caps['coco: ' + k], clean0_coco)
                r['imagenetv2']['clean cost'] = changed(clean0_inv2, caps['imagenetv2: ' + k])
            else:           # protection: perturbed vs clean captions of the SAME fine-tuned mini-VLM
                r['coco'][name] = coco_metrics(caps['coco: ' + k], caps['coco: %s: clean' % var])
                r['imagenetv2'][name] = changed(caps['imagenetv2: %s: clean' % var], caps['imagenetv2: ' + k])
            save()
        r['captions'] = True
        save()

    # ------------------------------------------------------------------ summary
    base_coco = json.load(open(out / 'coco_stage2.json'))['results'] if (out / 'coco_stage2.json').exists() else {}
    own = 'universal v (%s)' % args.encoder
    print('\n%s: fine-tuned encoder as a defense (COCO val, %d images; changed = vs clean captions of the same model)'
          % (args.encoder, len(x_coco)))
    print('%-34s %7s %9s %9s   %s' % ('condition', 'CIDEr', 'wrong', 'changed', 'ImageNetV2 changed@0.5 (all)'))
    if 'clean' in base_coco:
        b = base_coco['clean']
        print('%-34s %7.3f %8.1f%% %9s' % ('original encoder, clean', b['CIDEr'], b['wrong']['rate'], '—'))
    if own in base_coco:
        b = base_coco[own]
        print('%-34s %7.3f %8.1f%% %8.1f%%' % ('original encoder, fixed v', b['CIDEr'], b['wrong']['rate'], b['changed']['rate']))
    for var in args.variants:
        r = res['variants'].get(var, {})
        for name in ['clean', 'fixed v', 'adaptive v']:
            m = r.get('coco', {}).get(name)
            if not m:
                continue
            i = r['imagenetv2']['clean cost' if name == 'clean' else name]['changed@0.5']['all']
            print('%-34s %7.3f %8.1f%% %8.1f%%   %5.1f%% (%.1f-%.1f)%s' % (
                '%s, %s' % (var, name), m['CIDEr'], m['wrong']['rate'], m['changed']['rate'], i['rate'], *i['ci95'],
                '  <- cost: vs original model' if name == 'clean' else ''))
    print('>> Saved', file_res, 'and', file_caps)
