"""Transfer checks outside the SmolVLM family, plus a functional comparison of the 256M and 500M encoders.

usage: python transfer_other.py similarity
       python transfer_other.py classifiers --vs group_final final h6 ...
Writes results/similarity_256_500.json and results/transfer_other.json.
"""
import sys, argparse, time, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["similarity", "classifiers"])
ap.add_argument("--vs", nargs="*", default=[])
ap.add_argument("--n", type=int, default=500)
a = ap.parse_args()
log = Logger("transfer_other.log")
E = load_u8("eval")


@torch.no_grad()
def similarity():
    """Same clean images through the 256M and 500M encoders: how alike are their states, layer by layer?"""
    p256, m256 = load_vlm(M256); v256 = m256.model.vision_model
    _, m500 = load_vlm(M500); v500 = m500.model.vision_model
    acc, feats = {}, {}
    for i in range(0, 50, 10):
        px = p256.image_processor(images=[[to_pil(u)] for u in E[i:i + 10]], do_image_splitting=False,
                                  return_tensors="pt")["pixel_values"].flatten(0, 1).to(DEV)
        s1, s2 = encoder_states(v256, px), encoder_states(v500, px)
        for k in s1:
            acc.setdefault(k, []).append(tok_cos(s1[k], s2[k]).cpu())
            feats.setdefault(k, ([], []))
            feats[k][0].append(s1[k].flatten(0, 1)[::4].cpu()); feats[k][1].append(s2[k].flatten(0, 1)[::4].cpu())

    def cka(X, Y):  # linear CKA over tokens
        X = X - X.mean(0); Y = Y - Y.mean(0)
        return ((Y.T @ X).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()
    out = {k: {"token_cos": torch.cat(acc[k]).mean().item(),
               "cka": cka(torch.cat(feats[k][0]).to(DEV), torch.cat(feats[k][1]).to(DEV))} for k in acc}
    save_json("similarity_256_500.json", out)
    for k, r in out.items():
        log(f"256M vs 500M, clean images, {k:6s}: token cos {r['token_cos']:.3f}  CKA {r['cka']:.3f}")


@torch.no_grad()
def classifiers():
    import torchvision
    from transformers import AutoModel, AutoProcessor
    names = [l.split(" ", 1)[1].split(",")[0].strip() for l in
             open(ROOT.parent / "repo/Experiments ViT/python/data/synset_words.txt", encoding="utf-8")]
    y = torch.tensor(labels("eval")[:a.n])

    # --- models
    sp = AutoProcessor.from_pretrained("google/siglip-base-patch16-512")
    sm = AutoModel.from_pretrained("google/siglip-base-patch16-512").to(DEV).eval()
    tin = sp(text=[f"a photo of a {n}." for n in names], padding="max_length", max_length=64, return_tensors="pt").to(DEV)
    def tf_(ids):
        o = sm.get_text_features(input_ids=ids)
        return o if torch.is_tensor(o) else o.pooler_output
    T = F.normalize(torch.cat([tf_(tin["input_ids"][i:i + 100]) for i in range(0, 1000, 100)]), dim=-1)
    W = torchvision.models
    cls = {"ResNet-50": (W.resnet50(weights=W.ResNet50_Weights.IMAGENET1K_V2), W.ResNet50_Weights.IMAGENET1K_V2.transforms()),
           "ViT-B/16": (W.vit_b_16(weights=W.ViT_B_16_Weights.IMAGENET1K_V1), W.ViT_B_16_Weights.IMAGENET1K_V1.transforms())}
    for k in cls:
        cls[k][0].to(DEV).eval()

    def predict(pert):
        preds = {"SigLIP (Google)": [], **{k: [] for k in cls}}; scos = []
        for i in range(0, a.n, 25):
            pils = [to_pil(u, pert) for u in E[i:i + 25]]
            px = sp(images=pils, return_tensors="pt")["pixel_values"].to(DEV)
            vo = sm.vision_model(pixel_values=px)
            f = F.normalize(vo.pooler_output, dim=-1)
            preds["SigLIP (Google)"].append((f @ T.T).argmax(-1).cpu())
            scos.append(vo.last_hidden_state.cpu())
            for k, (m, tf) in cls.items():
                preds[k].append(m(torch.stack([tf(p) for p in pils]).to(DEV)).argmax(-1).cpu())
        return {k: torch.cat(v) for k, v in preds.items()}, torch.cat(scos)

    R = load_json("transfer_other.json", {})
    t = time.time()
    clean, sclean = predict(None)
    R["clean"] = {k: {"acc": (p == y).float().mean().item()} for k, p in clean.items()}
    log("clean accuracy:", {k: f"{v['acc']:.1%}" for k, v in R["clean"].items()}, f"({time.time() - t:.0f}s)")
    for name in ["random"] + a.vs:
        v = random_v() if name == "random" else load_v(name)
        pr, sv = predict(v)
        row = {}
        for k, p in pr.items():
            fool = (p != clean[k]).float().mean().item()
            row[k] = {"fooling_rate": fool, "ci95": wilson(int((p != clean[k]).sum()), a.n), "acc": (p == y).float().mean().item()}
        row["SigLIP (Google)"]["token_cos"] = tok_cos(sclean, sv).mean().item()
        R[name] = row; save_json("transfer_other.json", R)
        log(f"{name:12s} " + " | ".join(f"{k}: fool {r['fooling_rate']:.1%} acc {r['acc']:.1%}" for k, r in row.items())
            + f" | SigLIP token cos {row['SigLIP (Google)']['token_cos']:.3f}")


similarity() if a.what == "similarity" else classifiers()
