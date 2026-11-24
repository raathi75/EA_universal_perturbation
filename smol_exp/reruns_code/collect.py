"""Gather every result file into results/summary.json and print a compact table."""
import json, glob
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
J = lambda n: json.load(open(R / n, encoding="utf-8")) if (R / n).exists() else None
order = ["random", "group_final", "final", "h2", "h3", "h4", "h6", "h8", "h9", "h10", "h12", "multi", "multi_mid", "multi_all"]
S = {"vlm": {}, "train": {}}
for model in ["256M", "500M", "2.2B"]:
    for v in order:
        e = J(f"eval_{model}_{v}.json")
        if not e: continue
        row = {k: e[k] for k in ("off", "on", "layers") if k in e}
        S["vlm"].setdefault(model, {})[v] = row
for v in order:
    t = J(f"train_{v}.json")
    if t: S["train"][v] = {"states": t["states"], "loss": [h["loss"] for h in t["history"]], "minutes": t["history"][-1]["minutes"]}
S["similarity_256_500"] = J("similarity_256_500.json")
S["transfer_other"] = J("transfer_other.json")
S["robustness"] = J("robustness.json")
S["e0"] = J("e0_compare_group.json")
json.dump(S, open(R / "summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

f = lambda r, k: f"{r[k]['changed']:.1%}" if r and k in r else "  -  "
print(f"{'v':12s}" + "".join(f"{m + ' ' + s:>14s}" for m in S["vlm"] for s in ("off", "on")) + f"{'cos fin 256':>12s}{'cos fin 500':>12s}")
for v in order:
    cells = "".join(f"{f(S['vlm'][m].get(v), s):>14s}" for m in S["vlm"] for s in ("off", "on"))
    c = lambda m: f"{S['vlm'].get(m, {}).get(v, {}).get('layers', {}).get('final', float('nan')):.3f}"
    print(f"{v:12s}{cells}{c('256M'):>12s}{c('500M'):>12s}")
if S["transfer_other"]:
    print("\ntransfer (fooling rate):")
    for v, row in S["transfer_other"].items():
        if v == "clean": print("  clean acc", {k: f"{x['acc']:.1%}" for k, x in row.items()}); continue
        print(f"  {v:12s}", " | ".join(f"{k} {x['fooling_rate']:.1%}" for k, x in row.items()))
