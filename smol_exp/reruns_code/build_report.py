"""Inject every result into the report template: reads results/*.json, writes EA/bilan_reruns.html."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
J = lambda n: json.load(open(R / n, encoding="utf-8")) if (R / n).exists() else None

LABEL = {"random": "Bruit aléatoire ±ε", "group_final": "Groupe (sortie)", "final": "Sortie (ré-entraînée)",
         "multi_mid": "3 + 6 + 9", "multi": "3 + 6 + 9 + sortie", "multi_all": "1 → 12 + sortie"}
GROUP = {"random": "base", "group_final": "base", "final": "base", "multi_mid": "combo", "multi": "combo", "multi_all": "combo"}
order = ["random", "group_final", "final", "h2", "h3", "h4", "h6", "h8", "h9", "h10", "h12", "multi_mid", "multi", "multi_all"]


def cell(d):
    return None if not d else {"p": d["changed"], "lo": d["ci95"][0], "hi": d["ci95"][1], "n": d["n"]}


T = J("transfer_other.json")
rob = J("robustness.json") or {}
rows = []
for v in order:
    e256, e500, e22 = J(f"eval_256M_{v}.json") or {}, J(f"eval_500M_{v}.json") or {}, J(f"eval_2.2B_{v}.json") or {}
    t = T.get(v, {})
    tc = lambda k: None if k not in t else {"p": t[k]["fooling_rate"], "lo": t[k]["ci95"][0], "hi": t[k]["ci95"][1], "n": 500}
    rows.append({
        "id": v, "label": LABEL.get(v, f"Couche {v[1:]}"), "group": GROUP.get(v, "single"),
        "m256": cell(e256.get("off")), "on": cell(e256.get("on")), "m500": cell(e500.get("off")), "m22": cell(e22.get("off")),
        "siglip": tc("SigLIP (Google)"), "resnet": tc("ResNet-50"), "vit": tc("ViT-B/16"),
        "cos256": (e256.get("layers") or {}).get("final"), "cos500": (e500.get("layers") or {}).get("final"),
        "rob": {"after": rob[v]["after_preproc"]["final"], "real": rob[v]["real"]["final"]} if v in rob else None,
    })

sim = J("similarity_256_500.json")
W = json.load(open(HERE.parent / "weight_diff_per_layer.json", encoding="utf-8"))
DATA = {
    "rows": rows,
    "sim": [{"k": k, "cos": s["token_cos"], "cka": s["cka"]} for k, s in sim.items()],
    "weights": [{"k": k, "d": w["256M_vs_500M"], "g": w["google_vs_256M"]} for k, w in W.items() if k != "note"],
    "clean_acc": {k: x["acc"] for k, x in T["clean"].items()},
    "captions": J("caption_analysis.json"),
}
tpl = (HERE / "report_template.html").read_text(encoding="utf-8")
out = HERE.parent / "bilan_reruns.html"
out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(DATA, ensure_ascii=False)), encoding="utf-8")
print("wrote", out, len(out.read_text(encoding="utf-8")), "chars")
