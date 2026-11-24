"""Does a perturbation survive the real pipeline? Damage (token cosine) per encoder state in two settings:
'after_preproc' = v added to the 512x512 tensor the encoder sees (training path, no rounding/resampling)
'real'          = v saved in an 8-bit image and passed through SmolVLM's processor (already in eval_256M_*.json).
usage: python robustness.py group_final final h6 multi ..."""
import sys, torch
from common import *

E = load_u8("eval")[:50]
proc, model = load_vlm()
vis = model.model.vision_model
R = load_json("robustness.json", {})
with torch.no_grad():
    for name in sys.argv[1:]:
        v = random_v() if name == "random" else load_v(name)
        acc = {}
        for i in range(0, 50, 10):
            x = to01(E[i:i + 10])
            s0, s1 = encoder_states(vis, norm(x)), encoder_states(vis, norm((x + v).clamp(0, 1)))
            for k in s0:
                acc.setdefault(k, []).append(tok_cos(s0[k], s1[k]).cpu())
        after = {k: torch.cat(a).mean().item() for k, a in acc.items()}
        real = load_json(f"eval_256M_{name}.json", {}).get("layers") or layer_damage(proc, vis, E, v)
        R[name] = {"after_preproc": after, "real": real}
        save_json("robustness.json", R)
        print(f"{name:12s} final cos: after-preproc {after['final']:.3f} -> real pipeline {real['final']:.3f} | "
              f"h6: {after['h6']:.3f} -> {real['h6']:.3f}", flush=True)
