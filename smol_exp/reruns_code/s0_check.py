"""Sanity checks: data split, layer-wise forward == official forward, training speed and memory."""
import time, torch
from common import *

log = Logger("s0_check.log")
t = time.time()
B = load_u8("build"); E = load_u8("eval")
log("build", tuple(B.shape), "eval", tuple(E.shape), f"({time.time() - t:.0f}s)")
build, ev = files_split()
log("first build files:", build[:3], "| first eval files:", ev[:3])

proc, model = load_vlm()
vis = model.model.vision_model
x = norm(to01(E[:4]))
with torch.no_grad():
    ref = vis(pixel_values=x).last_hidden_state
    mine = encoder_states(vis, x)
log("max |mine.final - official| =", (mine["final"] - ref).abs().max().item(), "| states:", list(mine))
log("norm of residual stream per state:", {k: round(v.norm(dim=-1).mean().item(), 1) for k, v in mine.items()})

for target in ["final", "h6", "h3"]:
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    v = torch.zeros(1, 3, IMG, IMG, device=DEV, requires_grad=True)
    keys = TARGETS[target]; upto = None if "final" in keys else max(int(k[1:]) for k in keys)
    ts = []
    for it in range(4):
        xb = to01(B[it * 8:(it + 1) * 8]); torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                zc = encoder_states(vis, norm(xb), upto)
            za = encoder_states(vis, norm((xb + v).clamp(0, 1)), upto)
            loss = torch.stack([tok_cos(zc[k], za[k]).mean() for k in keys]).mean()
        loss.backward(); torch.cuda.synchronize(); ts.append(time.time() - t0)
    log(f"{target}: {sum(ts[1:]) / 3:.2f}s/step -> {sum(ts[1:]) / 3 * 63 * 10 / 60:.1f} min for 10 epochs | peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
