"""Evaluate perturbations on SmolVLM-256M (and other SmolVLMs) through the real pipeline.

usage: python evaluate.py --model 256M --vs group_final random final h6 ... [--off 200] [--on 100] [--layers 50]
Writes results/eval_{model}_{v}.json (stats + captions). Clean captions are cached per model/setting.
"""
import argparse, time, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="256M")
ap.add_argument("--vs", nargs="+", required=True)
ap.add_argument("--off", type=int, default=200)     # captions, splitting off
ap.add_argument("--on", type=int, default=0)        # captions, splitting on (default SmolVLM setting)
ap.add_argument("--layers", type=int, default=50)   # images for the layer-wise damage profile
ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()

MID = {"256M": M256, "500M": M500, "2.2B": M2B}[a.model]
dtype = torch.float16 if a.model == "2.2B" else torch.float32
log = Logger(f"eval_{a.model}.log")
E = load_u8("eval")
proc, model = load_vlm(MID, dtype)
vis = model.model.vision_model


def clean_caps(split, n):
    name = f"clean_{a.model}_{'on' if split else 'off'}_{n}.json"
    c = load_json(name)
    if c is None:
        t = time.time()
        c = captions(proc, model, E[:n], None, split, bs=a.bs if not split else max(1, a.bs // 4))
        save_json(name, c); log(f"clean captions {a.model} split={split} n={n}: {time.time() - t:.0f}s")
    return c


for name in a.vs:
    out_name = f"eval_{a.model}_{name}.json"
    R = load_json(out_name, {})
    v = random_v() if name == "random" else load_v(name)
    R["linf_255"] = v.abs().max().item() * 255
    t = time.time()
    if a.off and "off" not in R:
        c0 = clean_caps(False, a.off)
        c1 = captions(proc, model, E[:a.off], v, False, bs=a.bs)
        R["off"] = changed_stats(c0, c1); R["caps_off"] = c1
    if a.on and "on" not in R:
        c0 = clean_caps(True, a.on)
        c1 = captions(proc, model, E[:a.on], v, True, bs=max(1, a.bs // 4))
        R["on"] = changed_stats(c0, c1); R["caps_on"] = c1
    if a.layers and "layers" not in R:
        R["layers"] = layer_damage(proc, vis, E[:a.layers], v)
    save_json(out_name, R)
    s = f"{a.model} | {name:12s} | off {R['off']['changed']:.1%} (CI {R['off']['ci95'][0]:.1%}-{R['off']['ci95'][1]:.1%}, @0.7 {R['off']['changed@0.7']:.1%})" if "off" in R else f"{a.model} | {name}"
    if "on" in R: s += f" | on {R['on']['changed']:.1%} (n={R['on']['n']})"
    if "layers" in R: s += f" | cos final {R['layers']['final']:.3f}, h6 {R['layers']['h6']:.3f}"
    log(s + f"  ({time.time() - t:.0f}s)")
