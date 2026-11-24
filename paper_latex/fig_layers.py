"""Figure for Section 4.4 (middle-layer loss and transfer), same style as fig_cvpr.py.
Reads the re-run results in ../smol_exp/uap_smolvlm/reruns/ and writes fig/layers.pdf (+ .png)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "smol_exp" / "uap_smolvlm" / "reruns"
OUT = HERE / "fig"
BLUE = "#2a78d6"; BLUE_L = "#86b6ef"; GRAY = "#8a8984"; INK = "#0b0b0b"; INK2 = "#52514e"; LINE = "#c9c8c2"
serif = [f for f in ["Liberation Serif", "Times New Roman", "DejaVu Serif"] if f in font_manager.get_font_names()]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "mathtext.fontset": "stix",
                     "font.size": 8, "axes.edgecolor": LINE, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
J = lambda n: json.load(open(RES / n, encoding="utf-8"))


def rate(model, v):
    o = J(f"eval_{model}_{v}.json")["off"]
    return 100 * o["changed"], 100 * o["ci95"][0], 100 * o["ci95"][1]


def layers():
    singles = [2, 3, 4, 6, 8, 9, 10, 12]
    combos = [("multi_mid", "3+6+9"), ("multi", "3+6+9\n+out"), ("multi_all", "1–12\n+out")]
    fig = plt.figure(figsize=(3.3, 3.25))

    # (a) captions changed vs loss target
    ax = fig.add_axes([0.13, 0.565, 0.85, 0.36])
    xs = singles + [13]                                     # 13 = output (final layer + post-LN)
    xc = [15.2, 16.7, 18.2]
    for model, col, mk, lab in [("256M", BLUE, "o", "SmolVLM-256M (attacked)"), ("500M", BLUE_L, "s", "SmolVLM-500M (transfer)")]:
        r = [rate(model, f"h{k}") for k in singles] + [rate(model, "final")]
        ax.fill_between(xs, [x[1] for x in r], [x[2] for x in r], color=col, alpha=0.18, lw=0)
        ax.plot(xs, [x[0] for x in r], color=col, lw=1.6, marker=mk, ms=3.8, mec="white", mew=0.6, label=lab)
        for x, (v, _) in zip(xc, combos):
            p, lo, hi = rate(model, v)
            ax.plot([x, x], [lo, hi], color=col, lw=1.6, solid_capstyle="round")
            ax.plot([x], [p], marker=mk, color=col, ms=4.6 if v == "multi" else 3.8, mec="white", mew=0.6)
    ref = rate("256M", "group_final")[0]
    ax.axhline(ref, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.text(18.85, ref - 2.5, "reference $v$", fontsize=6.2, color=INK2, ha="right", va="top")
    p, _, _ = rate("256M", "multi")
    ax.text(16.7, p + 9.5, f"{p:.1f}", fontsize=6.4, color=INK, ha="center")
    ax.axvline(14.1, color=LINE, lw=0.6)
    ax.set_xticks(xs + xc)
    ax.set_xticklabels([str(k) for k in singles] + ["out"] + [c[1] for c in combos], fontsize=6.2)
    ax.set_xlim(1.3, 18.9); ax.set_ylim(0, 100)
    ax.set_ylabel("Captions changed (%)", fontsize=7); ax.tick_params(axis="y", labelsize=6.5)
    ax.text(7.6, -27, "one layer (loss on $h_k$)", fontsize=6.6, color=INK2, ha="center")
    ax.text(16.6, -27, "several layers", fontsize=6.6, color=INK2, ha="center")
    ax.grid(axis="y", color="#ecebe6", lw=0.5); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 0.97), ncol=2, frameon=False, fontsize=6.3, handletextpad=0.3, columnspacing=0.9)
    fig.text(0.01, 0.995, "(a) Loss target vs. captions changed (real pipeline, 200 images)", fontsize=7.2, color=INK, va="top")

    # (b) how similar the 256M and 500M encoders are, state by state
    S = J("similarity_256_500.json")
    ks = list(S); x = list(range(len(ks)))
    bx = fig.add_axes([0.13, 0.11, 0.85, 0.25])
    bx.axhline(0, color=LINE, lw=0.6)
    bx.plot(x, [S[k]["token_cos"] for k in ks], color=BLUE, lw=1.6, marker="o", ms=3.4, mec="white", mew=0.6)
    bx.plot(x, [S[k]["cka"] for k in ks], color=GRAY, lw=1.6, marker="o", ms=3.4, mec="white", mew=0.6)
    bx.text(4.3, S["h4"]["token_cos"] + 0.12, "token cosine", fontsize=6.4, color=INK)
    bx.text(6.2, S["h6"]["cka"] - 0.28, "linear CKA", fontsize=6.4, color=INK2)
    bx.set_xticks(x); bx.set_xticklabels(["emb"] + [str(i) for i in range(1, 13)] + ["out"], fontsize=6.2)
    bx.set_ylim(-0.75, 1.1); bx.set_yticks([-0.5, 0, 0.5, 1]); bx.tick_params(axis="y", labelsize=6.5)
    bx.set_xlabel("Encoder state (layer $k$)", fontsize=7, labelpad=1)
    bx.grid(axis="y", color="#ecebe6", lw=0.5); bx.set_axisbelow(True)
    for s in ["top", "right"]: bx.spines[s].set_visible(False)
    fig.text(0.01, 0.43, "(b) 256M vs. 500M encoders on the same clean images", fontsize=7.2, color=INK, va="top")

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "layers.pdf"); fig.savefig(OUT / "layers.png", dpi=200)


if __name__ == "__main__":
    layers()
