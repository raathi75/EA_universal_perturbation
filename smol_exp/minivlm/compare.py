"""Four pre-trainings, same mini-VLM recipe: tables of caption quality, attack and transfer (comparison.md)."""
import json

from common import ENCODERS, RESULTS_DIR

ENC = list(ENCODERS)
SMOL = 'universal v (SmolVLM-256M, 512 -> 224), transfer'


def load(enc, name):
    p = RESULTS_DIR / enc / name
    return json.load(open(p)) if p.exists() else None


def ci(r, key):
    if r is None or key not in r:
        return '—'
    a = r[key]
    return '%.1f%% (%.0f–%.0f)' % (a['rate'], *a['ci95'])


def vkey(src, target):
    return 'universal v (%s)' % target if src == target else 'universal v (%s), transfer' % src


if __name__ == '__main__':
    # SmolVLM-256M (smolvlm_reference.py) as an extra column of the first two tables, when computed
    REAL = ['smolvlm-256m', 'llava-ov-0.5b', 'qwen2-vl-2b', 'qwen2-vl-2b-224']   # real VLMs (real_vlm.py)
    COLS = ENC + [m for m in ['smolvlm'] + REAL if (RESULTS_DIR / m / 'coco_stage2.json').exists()]
    REAL = [m for m in REAL if m in COLS]
    coco = {e: load(e, 'coco_stage2.json') for e in COLS}
    inv2 = {e: load(e, 'eval_captions_stage2_xi10.json') for e in COLS}
    tok = {e: load(e, 'uap_xi10_tokens.json') for e in ENC}
    get = lambda d, e, k: (d[e] or {}).get('results', {}).get(k)
    L = ['# Mini-VLM: SigLIP, CLIP, I-JEPA, MAE (same LM, connector and data)', '',
         'ξ = 10/255, perturbations built on ImageNetV2, images rounded to 8 bits, 95% Wilson intervals.', '',
         '## COCO val2017, against the human captions', '',
         '(siglip, clip, ijepa, mae: mini-VLMs at 224 px; smolvlm: SmolVLM-256M with the perturbation of the report; '
         'smolvlm-256m, llava-ov-0.5b, qwen2-vl-2b (448 px), qwen2-vl-2b-224 (same model at 224 px): real VLMs, own perturbation run to convergence)', '',
         '| | ' + ' | '.join(COLS) + ' |', '|---|' + '---|' * len(COLS)]
    L.append('| CIDEr, clean (quality: must be close) | ' + ' | '.join(
        '%.3f' % get(coco, e, 'clean')['CIDEr'] if get(coco, e, 'clean') else '—' for e in COLS) + ' |')
    L.append('| CIDEr, own universal v | ' + ' | '.join(
        '%.3f' % get(coco, e, vkey(e, e))['CIDEr'] if get(coco, e, vkey(e, e)) else '—' for e in COLS) + ' |')
    for label, k in [('clean', lambda e: 'clean'), ('own universal v', lambda e: vkey(e, e)),
                     ('random ±ξ', lambda e: 'random ±xi')]:
        L.append('| wrong (far from every human caption), %s | ' % label + ' | '.join(
            ci(get(coco, e, k(e)), 'wrong') for e in COLS) + ' |')
    L.append('| changed vs clean caption, own universal v | ' + ' | '.join(
        ci(get(coco, e, vkey(e, e)), 'changed') for e in COLS) + ' |')
    L.append('| changed vs clean caption, random ±ξ | ' + ' | '.join(
        ci(get(coco, e, 'random ±xi'), 'changed') for e in COLS) + ' |')
    ckey = lambda e: 'universal v (%s, connector-output)' % e
    L.append('| CIDEr, v built on the connector outputs | ' + ' | '.join(
        '%.3f' % get(coco, e, ckey(e))['CIDEr'] if get(coco, e, ckey(e)) else '—' for e in COLS) + ' |')
    L.append('| wrong, v built on the connector outputs | ' + ' | '.join(
        ci(get(coco, e, ckey(e)), 'wrong') for e in COLS) + ' |')
    L.append('| changed vs clean caption, v built on the connector outputs | ' + ' | '.join(
        ci(get(coco, e, ckey(e)), 'changed') for e in COLS) + ' |')
    akey = lambda e: 'universal v (%s, caption loss)' % e
    L.append('| CIDEr, end-to-end v (likelihood of the clean captions) | ' + ' | '.join(
        '%.3f' % get(coco, e, akey(e))['CIDEr'] if get(coco, e, akey(e)) else '—' for e in COLS) + ' |')
    L.append('| wrong, end-to-end v | ' + ' | '.join(ci(get(coco, e, akey(e)), 'wrong') for e in COLS) + ' |')
    L.append('| changed vs clean caption, end-to-end v | ' + ' | '.join(
        ci(get(coco, e, akey(e)), 'changed') for e in COLS) + ' |')

    L += ['', '## ImageNetV2 held-out images of the SmolVLM report: captions changed (500 images)', '',
          '| | ' + ' | '.join(COLS) + ' |', '|---|' + '---|' * len(COLS)]
    for label, k in [('own universal v', lambda e: vkey(e, e)), ('random ±ξ', lambda e: 'random ±xi'),
                     ('v built on the connector outputs', lambda e: 'universal v (%s, connector-output)' % e),
                     ('end-to-end v (likelihood of the clean captions)', lambda e: 'universal v (%s, caption loss)' % e)]:
        L.append('| %s | ' % label + ' | '.join(
            ci((get(inv2, e, k(e)) or {}).get('changed@0.5'), 'all') if get(inv2, e, k(e)) else '—' for e in COLS) + ' |')

    L += ['', '## Transfer: captions changed on COCO (rows: perturbation built on, columns: model attacked)', '',
          '| v built on ↓ / model → | ' + ' | '.join(ENC) + ' |', '|---|' + '---|' * len(ENC)]
    for src in ENC + ['SmolVLM-256M']:
        L.append('| %s | ' % src + ' | '.join(
            ci(get(coco, e, SMOL if src == 'SmolVLM-256M' else vkey(src, e)), 'changed') for e in ENC) + ' |')

    L += ['', '## Token damage on the ImageNetV2 held-out images (mean token cosine, 1 = unchanged)', '',
          '| v built on ↓ / encoder → | ' + ' | '.join(ENC) + ' |', '|---|' + '---|' * len(ENC)]
    for src in ENC + ['SmolVLM-256M', 'random']:
        key = lambda e: SMOL if src == 'SmolVLM-256M' else 'random ±xi' if src == 'random' else vkey(src, e)
        L.append('| %s | ' % src + ' | '.join(
            '%.3f' % tok[e]['token_cos'][key(e)] if tok[e] and key(e) in tok[e]['token_cos'] else '—' for e in ENC) + ' |')

    # Real VLMs: transfers (real_vlm.py)
    if REAL:
        L += ['', '## Real VLMs: captions changed on COCO by each perturbation (rows: built on, columns: model attacked)', '',
              '| v built on ↓ / model → | ' + ' | '.join(REAL) + ' |', '|---|' + '---|' * len(REAL)]
        for src in ['own'] + ENC + ['smolvlm (report)'] + REAL:
            def key(m):
                if src == 'own':
                    return vkey(m, m)
                return None if src == m else 'universal v (%s), transfer' % src
            L.append('| %s | ' % src + ' | '.join(
                ci(get(coco, m, key(m)), 'changed') if key(m) else '(own)' for m in REAL) + ' |')

    # Defense: fine-tuned encoder (defense_finetune.py)
    dfs = {e: load(e, 'defense_ft.json') for e in ENC}
    if any(dfs.values()):
        L += ['', '## Defense: fine-tuning the encoder (as in the report, notebook 07), captions changed', '',
              'clean cost: clean captions vs those of the original mini-VLM; fixed v / adaptive v: perturbed vs clean '
              'captions of the same model. no-defense = the original encoder attacked by the same adaptive '
              'procedure (equal attack effort). COCO val (1,000 images) / ImageNetV2 held-out (500 images).', '',
              '| | ' + ' | '.join(ENC) + ' |', '|---|' + '---|' * len(ENC)]
        def dcell(e, var, name):
            r = ((dfs[e] or {}).get('variants', {}).get(var) or {})
            c = (r.get('coco') or {}).get(name)
            i = (r.get('imagenetv2') or {}).get('clean cost' if name == 'clean' else name)
            if not c or not i:
                return '—'
            return '%.1f%% / %.1f%%' % (c['changed']['rate'], i['changed@0.5']['all']['rate'])
        for var in ['no-defense', 'FT-fixed', 'FT-UAT']:
            for name, label in [('clean', 'clean cost'), ('fixed v', 'fixed v'), ('adaptive v', 'adaptive v')]:
                if var == 'no-defense' and name == 'clean':
                    continue
                L.append('| %s, %s | ' % (var, label) + ' | '.join(dcell(e, var, name) for e in ENC) + ' |')

    # Post-training with the encoder frozen vs unfrozen (run_unfrozen.sh)
    pairs = [e for e in ENC if (RESULTS_DIR / (e + '-unfrozen') / 'coco_stage2.json').exists()]
    if pairs:
        cols = [(e, m) for e in pairs for m in (e, e + '-unfrozen')]
        cc = {m: load(m, 'coco_stage2.json') for _, m in cols}
        ii = {m: load(m, 'eval_captions_stage2_xi10.json') for _, m in cols}
        L += ['', '## Post-training with the encoder frozen or unfrozen (stage 2), clean quality and robustness', '',
              'Cells: captions changed on COCO val / on the ImageNetV2 held-out images. fixed v = perturbation of the '
              'original encoder (for the frozen model it is also its own, recomputed one); adaptive v = recomputed '
              'against the model.', '',
              '| | ' + ' | '.join('%s %s' % (e, 'unfrozen' if m != e else 'frozen') for e, m in cols) + ' |',
              '|---|' + '---|' * len(cols)]
        def cell(m, key):
            c, i = get(cc, m, key), get(ii, m, key)
            if not c:
                return '—'
            return '%.1f%% / %s' % (c['changed']['rate'], '%.1f%%' % i['changed@0.5']['all']['rate'] if i else '—')
        L.append('| CIDEr, clean | ' + ' | '.join(
            '%.3f' % get(cc, m, 'clean')['CIDEr'] if get(cc, m, 'clean') else '—' for _, m in cols) + ' |')
        L.append('| wrong, clean (COCO) | ' + ' | '.join(ci(get(cc, m, 'clean'), 'wrong') for _, m in cols) + ' |')
        L.append('| fixed v (original encoder) | ' + ' | '.join(
            cell(m, vkey(e, e) if m == e else 'universal v (%s), transfer' % e) for e, m in cols) + ' |')
        L.append('| adaptive v (recomputed, tokens) | ' + ' | '.join(cell(m, 'universal v (%s)' % m) for _, m in cols) + ' |')
        L.append('| adaptive v (recomputed, connector outputs) | ' + ' | '.join(
            cell(m, 'universal v (%s, connector-output)' % m) for _, m in cols) + ' |')
        L.append('| random ±ξ | ' + ' | '.join(cell(m, 'random ±xi') for _, m in cols) + ' |')
        L.append('| CIDEr, adaptive v (tokens) | ' + ' | '.join(
            '%.3f' % get(cc, m, 'universal v (%s)' % m)['CIDEr'] if get(cc, m, 'universal v (%s)' % m) else '—'
            for _, m in cols) + ' |')

    # Convergence of the attacks
    L += ['', '## Convergence of the attacks (epoch kept / epochs run, score on the build images: feature cosine, '
              'or caption likelihood for the end-to-end v)', '',
          '| | ' + ' | '.join(COLS) + ' |', '|---|' + '---|' * len(COLS)]
    def conv(rec):
        if not rec or 'best_epoch' not in rec:
            return '—'
        return '%d / %d, %.3f%s' % (rec['best_epoch'], rec['epoch'], rec['best'], '' if rec.get('converged') else ' (not converged)')
    for label, f in [('token v', 'uap_xi10_history.json'), ('connector-output v', 'uap_conn_xi10_history.json'),
                     ('end-to-end v', 'uap_cap_xi10_history.json')]:
        L.append('| %s | ' % label + ' | '.join(conv(load(e, f)) for e in COLS) + ' |')
    for var in ['no-defense', 'FT-fixed', 'FT-UAT']:
        L.append('| adaptive v, %s | ' % var + ' | '.join(
            conv(((dfs.get(e) or {}).get('variants', {}).get(var) or {}).get('adaptive_history')) for e in COLS) + ' |')

    # Variance of the attack (seeds of the own token v) and of the random control (draws) — run_complete.sh
    def spread(e, prefix, d):
        rows = [(k, r) for k, r in ((d[e] or {}).get('results', {})).items() if (k == prefix or k.startswith(prefix + ', ')) and 'transfer' not in k]
        if len(rows) < 2:
            return '—'
        vals = [(r['changed@0.5']['all'] if 'changed@0.5' in r else r['changed'])['rate'] for _, r in rows]
        m = sum(vals) / len(vals)
        return '%.1f%% [%.1f–%.1f] (n=%d)' % (m, min(vals), max(vals), len(vals))
    L += ['', '## Variance: own token v over attack seeds, random ±ξ over draws (mean [min–max], captions changed)', '',
          '| | ' + ' | '.join(COLS) + ' |', '|---|' + '---|' * len(COLS)]
    for label, d, pre in [('own v, seeds, COCO', coco, None), ('own v, seeds, ImageNetV2', inv2, None),
                          ('random ±ξ, draws, COCO', coco, 'random ±xi'), ('random ±ξ, draws, ImageNetV2', inv2, 'random ±xi')]:
        L.append('| %s | ' % label + ' | '.join(spread(e, pre or 'universal v (%s)' % e, d) for e in COLS) + ' |')
    mech = {e: load(e, 'defense_mechanism.json') for e in ENC}
    if any(mech.values()):
        L += ['', '## Defense mechanism: token vs connector-output cosine on the held-out images (1 = unchanged)', '',
              '| | ' + ' | '.join(ENC) + ' |', '|---|' + '---|' * len(ENC)]
        for var in ['no-defense', 'FT-fixed', 'FT-UAT']:
            for v in ['fixed v', 'adaptive v']:
                L.append('| %s, %s (token / connector) | ' % (var, v) + ' | '.join(
                    '%.3f / %.3f' % (r['token_cos'], r['connector_cos'])
                    if (r := ((mech[e] or {}).get('variants', {}).get(var) or {}).get(v)) else '—' for e in ENC) + ' |')

    # ImageNet accuracy read from the captions (imagenet_accuracy.py)
    acc = json.load(open(RESULTS_DIR / 'imagenet_accuracy.json')) if (RESULTS_DIR / 'imagenet_accuracy.json').exists() else {}
    if acc:
        L += ['', '## ImageNet accuracy read from the captions (500 ImageNetV2 held-out images, top-1 / top-5, %)', '',
              'Each caption is matched (MiniLM) with the 1000 class names; captions are generic, so compare columns '
              'and rows, not with a classifier.', '',
              '| | ' + ' | '.join(COLS) + ' |', '|---|' + '---|' * len(COLS)]
        def a(m, k):
            r = acc.get(m, {}).get(k)
            return '%.1f / %.1f' % (r['top1'], r['top5']) if r else '—'
        rows = [('clean', lambda m: 'clean'), ('own universal v', lambda m: vkey(m, m) if m in ENC else 'universal v (%s)' % m),
                ('end-to-end v', lambda m: 'universal v (%s, caption loss)' % m), ('random ±ξ', lambda m: 'random ±xi')]
        rows += [('%s, %s' % (var, c), (lambda var, c: lambda m: 'defense | %s: %s' % (var, c))(var, c))
                 for var in ['FT-fixed', 'FT-UAT'] for c in ['clean', 'fixed v', 'adaptive v']]
        rows.insert(3, ('no-defense, adaptive v', lambda m: 'defense | no-defense: adaptive v'))
        for label, k in rows:
            L.append('| %s | ' % label + ' | '.join(a(m, k(m)) for m in COLS) + ' |')

    text = '\n'.join(L)
    print(text)
    (RESULTS_DIR / 'comparison.md').write_text(text + '\n')
    print('\n>> Saved', RESULTS_DIR / 'comparison.md')
