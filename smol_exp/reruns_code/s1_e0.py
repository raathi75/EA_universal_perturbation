"""E0: do our captions match the group's? Compare with their saved captions (week3_finetune_and_rrp.json)."""
import json
from common import *

G = json.load(open(ROOT.parent / "repo/smol_exp/uap_smolvlm/week3_finetune_and_rrp.json", encoding="utf-8"))["captions"]
mine_clean = load_json("clean_256M_off_200.json")
mine_v = load_json("eval_256M_group_final.json")["caps_off"]
same_clean = sum(a == b for a, b in zip(mine_clean, G["clean, original encoder"]))
same_v = sum(a == b for a, b in zip(mine_v, G["fixed v, no defense"]))
their = changed_stats(G["clean, original encoder"], G["fixed v, no defense"])
out = {"identical_clean_captions": same_clean, "identical_perturbed_captions": same_v, "n": len(mine_clean),
       "their_rate_recomputed": their}
save_json("e0_compare_group.json", out)
print(out)
for i in range(200):
    if mine_clean[i] != G["clean, original encoder"][i]:
        print("diff", i, "|", mine_clean[i], "|", G["clean, original encoder"][i]); break
