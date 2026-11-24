"""Test of the mechanism proposed for the failure of the fine-tuning defense (RAPPORT.md, IV.3): the defense protects
the encoder TOKENS but pushes the adaptive attacker towards directions the CONNECTOR uses.

For no-defense / FT-fixed / FT-UAT, on the 500 ImageNetV2 held-out images, with the fixed v and the adaptive v:
  - token cosine (clean vs perturbed tokens of the same encoder; already in defense_ft.json, recomputed here),
  - connector-output cosine (clean vs perturbed outputs of the trained connector, what the language model receives).
Prediction of the hypothesis: under the adaptive v, FT has a HIGHER token cosine than no-defense but a connector
cosine as LOW or lower (siglip). Needs the fine-tuned encoders: run on the machine that trained them.
  python defense_mechanism.py --encoder siglip          -> results/<encoder>/defense_mechanism.json
"""
import argparse
import json

import torch

from common import DATA_DIR, ENCODERS, LM_ID, Connector, Encoder, feature_cosine, features_of, get_device, \
    imagenetv2_images, load_perturbation, results_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Token vs connector-output damage under the defense')
    parser.add_argument('--encoder', required=True, choices=list(ENCODERS))
    args = parser.parse_args()

    from transformers import AutoConfig
    device = get_device()
    out = results_dir(args.encoder)
    encoder = Encoder(args.encoder).to(device)
    original = {k: t.detach().cpu().clone() for k, t in encoder.state_dict().items()}
    connector = Connector(encoder.dim, AutoConfig.from_pretrained(LM_ID).hidden_size)
    connector.load_state_dict(torch.load(out / 'connector_stage2.pt', map_location='cpu'))
    connector = connector.to(device).eval()
    for p in connector.parameters():
        p.requires_grad_(False)
    _, eval_u8, _, _ = imagenetv2_images()
    v_fixed = load_perturbation(out / 'uap_xi10.pt', device)
    conn_fn = lambda x: connector(encoder(x).float())

    res = {'encoder': args.encoder, 'n_images': len(eval_u8), 'variants': {}}
    # a machine without the defense checkpoints must not erase variants measured elsewhere
    previous = {}
    if (out / 'defense_mechanism.json').exists():
        previous = json.load(open(out / 'defense_mechanism.json')).get('variants', {})
    for var in ['no-defense', 'FT-fixed', 'FT-UAT']:
        path_ft = DATA_DIR / 'defense' / ('%s_%s.pt' % (args.encoder, var))
        if var == 'no-defense':
            encoder.load_state_dict(original)
        elif path_ft.exists():
            encoder.load_state_dict(torch.load(path_ft, map_location='cpu'))
        else:
            print('!! %s not found on this machine: run where the defense was trained' % path_ft)
            if var in previous:
                print('   keeping the %s values already in defense_mechanism.json' % var)
                res['variants'][var] = previous[var]
            continue
        v_ad = load_perturbation(out / ('adaptive_xi10_%s.pt' % var), device)
        clean_t = features_of(encoder, eval_u8, device)
        clean_c = features_of(conn_fn, eval_u8, device)
        r = {}
        for name, v in [('fixed v', v_fixed), ('adaptive v', v_ad)]:
            r[name] = {'token_cos': feature_cosine(encoder, eval_u8, device, clean_t, v).mean().item(),
                       'connector_cos': feature_cosine(conn_fn, eval_u8, device, clean_c, v).mean().item()}
        res['variants'][var] = r
        print('%-11s fixed v: token %.3f conn %.3f | adaptive v: token %.3f conn %.3f' % (
            var, r['fixed v']['token_cos'], r['fixed v']['connector_cos'],
            r['adaptive v']['token_cos'], r['adaptive v']['connector_cos']))
    json.dump(res, open(out / 'defense_mechanism.json', 'w'), indent=1)
    print('>> Saved', out / 'defense_mechanism.json')
