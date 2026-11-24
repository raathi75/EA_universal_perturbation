"""Label-free universal perturbation on a mini-VLM, built on the 500 ImageNetV2 build images and run to
CONVERGENCE (common.converge_uap: same loss and optimiser as the SmolVLM notebooks, mean cosine between clean and
perturbed features, Adam lr 1/255, batch 8, l_inf projection; the best v on the build images is kept, the step is
halved on plateaus, at least the 10 epochs of the report).

  --target tokens     (default, as in the report) the 256 tokens of the frozen encoder -> uap_xi10.pt
                      only needs the encoder, so it can run before the connector is trained
  --target connector  the 64 connector outputs, i.e. what the language model actually receives -> uap_conn_xi10.pt
                      needs the trained connector (stage 2)

Then reports the damage on the 500 held-out images: for tokens, the token cosine of this perturbation, of random
noise of the same budget, and of the perturbations of the other encoders and of SmolVLM-256M (transfer);
for connector, the cosine of the connector outputs and of the tokens, for this v and for the token v.
"""
import argparse
import json

import torch

from common import (MODELS, ENCODERS, HERE, IMG_SIZE, SEED, XI, Connector, Encoder, converge_uap, feature_cosine, seed_start, seed_tag, features_of, get_device,
                    imagenetv2_images, LM_ID, load_minivlm, load_perturbation, other_encoders, results_dir, token_cosine)


def damage(feat_fn, images_u8, v, device, clean=None):
    clean = features_of(feat_fn, images_u8, device) if clean is None else clean
    return feature_cosine(feat_fn, images_u8, device, clean, v).mean().item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universal perturbation on a mini-VLM, to convergence (label-free)')
    parser.add_argument('--encoder', required=True, choices=MODELS)
    parser.add_argument('--target', default='tokens', choices=['tokens', 'connector', 'caption'])
    parser.add_argument('--xi', type=float, default=XI * 255, help='l_inf budget on the [0, 255] scale')
    parser.add_argument('--lr', type=float, default=1, help='initial Adam step on the [0, 255] scale')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0,
                        help='0 = the original run; k > 0: random start in the ball and another image order -> *_s<k>.pt '
                             '(variance of the attack)')
    args = parser.parse_args()

    device = get_device()
    out = results_dir(args.encoder)
    tag = {'tokens': 'uap_xi%g', 'connector': 'uap_conn_xi%g', 'caption': 'uap_cap_xi%g'}[args.target] % args.xi
    tag += seed_tag(args.seed)
    file_v, ckpt_path = out / (tag + '.pt'), out / (tag + '_ckpt.pt')
    xi = args.xi / 255

    torch.manual_seed(SEED + args.seed)
    score_fn = None
    if args.target == 'caption':
        # END-TO-END attack: likelihood of the model's own clean caption of each build image (exp(-mean cross-entropy),
        # 1 = unchanged), through encoder, connector and language model (LoRA): the strongest test of robustness
        vlm = load_minivlm(args.encoder, device, stage=2)
        for p in vlm.parameters():
            p.requires_grad_(False)
        encoder = vlm.encoder
    else:
        encoder = Encoder(args.encoder).to(device)
    feat_tokens = encoder
    if args.target == 'caption':
        feat_fn = feat_tokens
    elif args.target == 'connector':
        from transformers import AutoConfig
        lm_dim = AutoConfig.from_pretrained(LM_ID).hidden_size
        connector = Connector(encoder.dim, lm_dim)
        connector.load_state_dict(torch.load(out / 'connector_stage2.pt', map_location='cpu'))
        connector = connector.to(device).eval()
        for p in connector.parameters():
            p.requires_grad_(False)
        feat_fn = lambda x: connector(encoder(x))
    else:
        feat_fn = feat_tokens
    build_u8, eval_u8, build_files, eval_files = imagenetv2_images()
    print('>> %s, target %s: build %s, held-out %s' % (args.encoder, args.target, tuple(build_u8.shape), tuple(eval_u8.shape)))

    if args.target == 'caption' and not file_v.exists():
        f_caps = out / 'build_captions.json'
        if f_caps.exists():
            build_caps = json.load(open(f_caps))
        else:
            print('>> clean captions of the %d build images (target of the end-to-end attack)' % len(build_u8))
            build_caps = vlm.caption(build_u8.to(device).float() / 255)
            json.dump(build_caps, open(f_caps, 'w'), indent=1)
        score_fn = lambda x, idx: torch.exp(-vlm.caption_loss(
            encoder(x).float(), [build_caps[i] for i in idx.tolist()], reduction='none'))
    if not file_v.exists():
        v, rec = converge_uap(feat_fn, build_u8, device, v0=seed_start(args.seed, IMG_SIZE, xi), seed=args.seed,
                              xi=xi, lr=args.lr / 255, batch_size=args.batch_size,
                              ckpt_path=ckpt_path, max_epochs=args.max_epochs, min_epochs=args.min_epochs,
                              name='%s %s' % (args.encoder, args.target), score_fn=score_fn,
                              eval_bs=args.batch_size if score_fn else 32)
        torch.save(v, file_v)
        json.dump(rec, open(out / (tag + '_history.json'), 'w'), indent=1)
        ckpt_path.unlink(missing_ok=True)

    # Damage on the held-out images (perturbed images rounded to 8 bits)
    torch.manual_seed(SEED)
    res = {'encoder': args.encoder, 'target': args.target, 'xi': args.xi, 'n_eval': len(eval_u8), 'eval_files': eval_files}
    v = load_perturbation(file_v, device)
    if args.target == 'caption':   # how much the end-to-end v damages the tokens, next to the token v
        perts = {'caption-loss v': v}
        tok_v = out / ('uap_xi%g.pt' % args.xi)
        if tok_v.exists():
            perts['token v (uap_xi%g.pt)' % args.xi] = load_perturbation(tok_v, device)
        clean = features_of(feat_tokens, eval_u8, device)
        res['token_cos'] = {}
        for name, p in perts.items():
            res['token_cos'][name] = damage(feat_tokens, eval_u8, p, device, clean)
            print('%-40s token cos %.3f' % (name, res['token_cos'][name]))
    elif args.target == 'tokens':
        perts = {'universal v (%s)' % args.encoder: v,
                 'random ±xi': xi * torch.sign(torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device))}
        for o in other_encoders(args.encoder):
            other = results_dir(o) / (tag + '.pt')
            if other.exists():
                perts['universal v (%s), transfer' % o] = load_perturbation(other, device)
        smol = HERE.parent / 'uap_smolvlm' / ('uap_smolvlm256m_linf%g.npy' % args.xi)
        if smol.exists():
            perts['universal v (SmolVLM-256M, 512 -> 224), transfer'] = load_perturbation(smol, device)
        clean = features_of(feat_tokens, eval_u8, device)
        res['token_cos'] = {}
        print('\n%-55s %10s' % ('perturbation (held-out, %d images)' % len(eval_u8), 'token cos'))
        for name, p in perts.items():
            res['token_cos'][name] = damage(feat_tokens, eval_u8, p, device, clean)
            print('%-55s %10.3f' % (name, res['token_cos'][name]))
    else:
        perts = {'connector-output v': v}
        tok_v = out / ('uap_xi%g.pt' % args.xi)
        if tok_v.exists():
            perts['token v (uap_xi%g.pt)' % args.xi] = load_perturbation(tok_v, device)
        perts['random ±xi'] = xi * torch.sign(torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device))
        clean_c, clean_t = features_of(feat_fn, eval_u8, device), features_of(feat_tokens, eval_u8, device)
        res['connector_cos'], res['token_cos'] = {}, {}
        print('\n%-40s %15s %10s' % ('perturbation (held-out, %d images)' % len(eval_u8), 'connector cos', 'token cos'))
        for name, p in perts.items():
            res['connector_cos'][name] = damage(feat_fn, eval_u8, p, device, clean_c)
            res['token_cos'][name] = damage(feat_tokens, eval_u8, p, device, clean_t)
            print('%-40s %15.3f %10.3f' % (name, res['connector_cos'][name], res['token_cos'][name]))
    json.dump(res, open(out / (tag + '_tokens.json'), 'w'), indent=1)
    print('>> Saved', file_v, 'and', out / (tag + '_tokens.json'))
