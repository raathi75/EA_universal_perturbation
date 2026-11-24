"""Train one universal perturbation per target: python train.py final h6 multi ..."""
import sys, json
from common import *

log = Logger("train.log")
B = load_u8("build")
proc, model = load_vlm()
vis = model.model.vision_model
for target in sys.argv[1:]:
    if (PERT / f"v_{target}.npy").exists() and (RES / f"train_{target}.json").exists():
        log(f"[{target}] already trained, skipping"); continue
    if target.startswith("h") and target not in TARGETS:
        TARGETS[target] = [target]
    v, hist = train_uap(vis, B, target, log=log)
    save_json(f"train_{target}.json", {"target": target, "states": TARGETS[target], "history": hist})
