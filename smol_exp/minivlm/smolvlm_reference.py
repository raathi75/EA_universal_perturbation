"""SmolVLM-256M (the model of the report) next to the mini-VLMs, with the same images and metrics.

A. Was SmolVLM's vision encoder changed by HuggingFace's training? Its weights are compared with those of the
   original Google encoder it starts from (SigLIP-B/16-512): relative difference of the weights, and token cosine
   between the two encoders on 100 held-out ImageNetV2 images (1 = identical features).
B. Caption quality and attack on the SAME 1,000 COCO val images as the mini-VLMs (CIDEr, wrong vs the human captions,
   changed), and on the 500 ImageNetV2 held-out images (changed): clean, the report's universal v, random noise.
   Protocol of the notebooks: 512 x 512 (Resize + CenterCrop, same field of view as the 224 crops of the mini-VLMs),
   one tile (no image splitting), same prompt, greedy decoding, 40 tokens; perturbed images rounded to 8 bits.

Results go to results/smolvlm/ with the same file names and keys as the mini-VLMs, so compare.py adds a column.
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from common import (CAPTION_SIM_CHANGED, DATA_DIR, HERE, MAX_NEW_TOKENS, PROMPT, SEED, XI, CocoCaptions, TextSim,
                    get_device, imagenetv2_split, quantize, results_dir, token_cosine)
from eval_coco import rate, tokenize
from evaluate_captions import rates

SMOL_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct'
SIGLIP_ORIGINAL = 'google/siglip-base-patch16-512'
SIZE = 512
PRE512 = transforms.Compose([transforms.Resize(SIZE), transforms.CenterCrop(SIZE), transforms.PILToTensor()])


def load512(paths):
    return torch.stack([PRE512(Image.open(p).convert('RGB')) for p in paths])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SmolVLM-256M as a reference for the mini-VLMs')
    parser.add_argument('--n_coco', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor, SiglipVisionModel

    device = get_device()
    out = results_dir('smolvlm')
    processor = AutoProcessor.from_pretrained(SMOL_ID)
    model = AutoModelForImageTextToText.from_pretrained(SMOL_ID, torch_dtype=torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vis = model.model.vision_model
    ip = processor.image_processor
    ip.do_image_splitting = False   # one 512 x 512 view (recent transformers ignore it as a call argument)
    mean = torch.tensor(ip.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device=device).view(1, 3, 1, 1)
    norm = lambda x01: (x01 - mean) / std

    _, eval_files = imagenetv2_split()
    inv2_u8 = load512(eval_files)

    # ------------------------------------------------------------------ A. was the encoder fine-tuned?
    file_w = out / 'encoder_vs_original.json'
    if not file_w.exists():
        orig = SiglipVisionModel.from_pretrained(SIGLIP_ORIGINAL).to(device).eval()
        o_params = dict(orig.named_parameters())
        diffs, n_missing = {}, 0
        for k, p in vis.named_parameters():
            q = o_params.get('vision_model.' + k)
            if q is None or q.shape != p.shape:
                n_missing += 1
                continue
            diffs[k] = ((p - q).norm() / q.norm().clamp_min(1e-12)).item()
        with torch.no_grad():
            x = inv2_u8[:100].to(device).float() / 255
            tc = torch.cat([token_cosine(vis(pixel_values=norm(x[i:i + 10])).last_hidden_state,
                                         orig(pixel_values=norm(x[i:i + 10])).last_hidden_state).cpu()
                            for i in range(0, 100, 10)])
        w = {'original': SIGLIP_ORIGINAL, 'n_compared': len(diffs), 'n_not_matched': n_missing,
             'mean_relative_weight_change': float(np.mean(list(diffs.values()))) if diffs else None,
             'max_relative_weight_change': max(diffs.values()) if diffs else None,
             'identical_tensors': sum(d == 0 for d in diffs.values()),
             'token_cos_smolvlm_vs_original_100_images': tc.mean().item(),
             'per_parameter': diffs}
        json.dump(w, open(file_w, 'w'), indent=1)
        del orig
        torch.cuda.empty_cache()
    w = json.load(open(file_w))
    print('\nA. SmolVLM encoder vs the original %s: %d tensors compared (%d not matched), %d identical' % (
        w['original'], w['n_compared'], w['n_not_matched'], w['identical_tensors']))
    if w['n_compared']:
        print('   mean relative weight change %.4f (max %.4f)' % (w['mean_relative_weight_change'], w['max_relative_weight_change']))
    print('   token cosine between the two encoders on 100 images: %.4f (1 = same features)' % (
        w['token_cos_smolvlm_vs_original_100_images']))

    # ------------------------------------------------------------------ B. captions
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    base = processor(text=prompt_text, images=[Image.fromarray(np.zeros((SIZE, SIZE, 3), dtype=np.uint8))],
                     return_tensors='pt')
    assert tuple(base['pixel_values'].shape[-3:]) == (3, SIZE, SIZE) and base['pixel_values'].shape[1] == 1, \
        'processor output %s, expected 1 view of 3 x %d x %d' % (tuple(base['pixel_values'].shape), SIZE, SIZE)

    @torch.no_grad()
    def caption(images_u8, v=None):
        outs = []
        for i in range(0, len(images_u8), args.batch_size):
            x = images_u8[i:i + args.batch_size].to(device).float() / 255
            if v is not None:
                x = quantize(x + v)
            b = len(x)
            inp = {k: t.repeat(b, *[1] * (t.dim() - 1)).to(device) for k, t in base.items() if k != 'pixel_values'}
            inp['pixel_values'] = norm(x).unsqueeze(1)
            gen = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            outs += [o.strip() for o in processor.batch_decode(gen[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)]
        return outs

    from pycocoevalcap.cider.cider import Cider
    sim_fn = TextSim(device)
    coco = CocoCaptions('val2017', max_images=args.n_coco)
    coco_u8 = load512([p for p, _ in coco.items])
    refs = [r for _, r in coco.items]
    gts = {i: [tokenize(c) for c in r] for i, r in enumerate(refs)}
    ref_emb = sim_fn.st.encode([c for r in refs for c in r], convert_to_tensor=True, normalize_embeddings=True)
    owner = torch.tensor([i for i, r in enumerate(refs) for _ in r], device=ref_emb.device)

    file_coco, file_coco_caps = out / 'coco_stage2.json', out / 'coco_captions_stage2.json'
    file_inv2, file_inv2_caps = out / 'eval_captions_stage2_xi10.json', out / 'captions_stage2_xi10.json'
    res_c = json.load(open(file_coco)) if file_coco.exists() else {'model': SMOL_ID, 'results': {}}
    caps_c = json.load(open(file_coco_caps)) if file_coco_caps.exists() else {}
    res_i = json.load(open(file_inv2)) if file_inv2.exists() else {'model': SMOL_ID, 'results': {}}
    caps_i = json.load(open(file_inv2_caps)) if file_inv2_caps.exists() else {}
    res_c['n_images'], res_i['n_images'] = len(coco_u8), len(inv2_u8)

    torch.manual_seed(SEED)
    v_report = torch.from_numpy(np.load(HERE.parent / 'uap_smolvlm' / 'uap_smolvlm256m_linf10.npy')).float()
    conds = {'clean': None,
             'universal v (smolvlm)': v_report.reshape(1, 3, SIZE, SIZE).to(device),
             'random ±xi': XI * torch.sign(torch.randn(1, 3, SIZE, SIZE, device=device))}
    for name, v in conds.items():
        if name not in res_c['results']:
            print('>> COCO val, %d images: %s' % (len(coco_u8), name))
            caps_c[name] = caption(coco_u8, v)
            cider, _ = Cider().compute_score(gts, {i: [tokenize(c)] for i, c in enumerate(caps_c[name])})
            e = sim_fn.st.encode(caps_c[name], convert_to_tensor=True, normalize_embeddings=True)
            s = (e[owner] * ref_emb).sum(-1)
            best = torch.full((len(e),), -1.0, device=s.device).scatter_reduce(0, owner, s, reduce='amax').cpu()
            r = {'CIDEr': cider, 'mean_best_human_sim': best.mean().item(), 'wrong': rate(best < CAPTION_SIM_CHANGED)}
            if v is not None:
                r['changed'] = rate(sim_fn(caps_c['clean'], caps_c[name]) < CAPTION_SIM_CHANGED)
            res_c['results'][name] = r
            json.dump(caps_c, open(file_coco_caps, 'w'), indent=1)
            json.dump(res_c, open(file_coco, 'w'), indent=1)
        if name not in res_i['results'] and (v is not None or 'clean' not in caps_i):
            print('>> ImageNetV2 held-out, %d images: %s' % (len(inv2_u8), name))
            caps_i[name] = caption(inv2_u8, v)
            if v is not None:
                sim = sim_fn(caps_i['clean'], caps_i[name])
                res_i['results'][name] = {'mean_caption_sim': sim.mean().item(),
                                          'changed@0.5': rates(sim, CAPTION_SIM_CHANGED), 'changed@0.7': rates(sim, 0.7)}
            json.dump(caps_i, open(file_inv2_caps, 'w'), indent=1)
            json.dump(res_i, open(file_inv2, 'w'), indent=1)

    print('\nB. SmolVLM-256M, COCO val (%d images) and ImageNetV2 held-out (%d images), xi = 10/255' % (
        len(coco_u8), len(inv2_u8)))
    print('%-24s %7s %9s %9s   %s' % ('condition', 'CIDEr', 'wrong', 'changed', 'ImageNetV2 changed@0.5 (first 200 | all)'))
    for name, r in res_c['results'].items():
        i = res_i['results'].get(name, {}).get('changed@0.5')
        print('%-24s %7.3f %8.1f%% %9s   %s' % (
            name, r['CIDEr'], r['wrong']['rate'], '%.1f%%' % r['changed']['rate'] if 'changed' in r else '—',
            '%.1f%% | %.1f%%' % (i['first200']['rate'], i['all']['rate']) if i else '—'))
    print('\nExamples (human / SmolVLM clean / SmolVLM with v):')
    for k in range(3):
        print('  human:', refs[k][0], '\n  clean:', caps_c['clean'][k], '\n  v    :', caps_c['universal v (smolvlm)'][k], '\n')
    print('>> Saved', out)
