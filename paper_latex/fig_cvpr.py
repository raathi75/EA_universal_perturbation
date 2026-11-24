"""Figures for the CVPR-format workshop paper. All numbers come from the project log / Nov 12 report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from math import sqrt

OUT = "/home/claude/paper_cvpr/fig/"
BLUE = "#2a78d6"; BLUE_L = "#86b6ef"; GRAY = "#8a8984"; INK = "#0b0b0b"; INK2 = "#52514e"; LINE = "#c9c8c2"; RED = "#c0392b"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.edgecolor": LINE, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

# ---------------- Teaser: clean vs perturbed, three scenes ----------------
def teaser():
    clean = ["a bird on the branch of a tree.", "two dogs.", "a machine and some objects."]
    off = ["a person wearing a white shirt", "a person wearing a white shirt", "a person wearing a white shirt"]
    on = ["a collage of images.", "a picture of a grass.", "a vehicle on the road."]
    fig = plt.figure(figsize=(6.9, 2.05))
    W = 1 / 3
    for j in range(3):
        for i, tag in enumerate(["Clean", "+ universal $v$ ($\\epsilon=10/255$)"]):
            ax = fig.add_axes([j * W + 0.006 + i * 0.163, 0.29, 0.158, 0.66])
            ax.imshow(Image.open(f"/home/claude/crops/pipe_{i}{j}.png")); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_title(tag, fontsize=7, color=INK2, pad=2)
        x = j * W + 0.008
        fig.text(x, 0.255, "Clean: ", fontsize=6.8, color=INK2, weight="bold", va="top")
        fig.text(x + 0.064, 0.255, f"“{clean[j]}”", fontsize=6.8, color=INK, va="top")
        fig.text(x, 0.16, "$v$, 1 tile: ", fontsize=6.8, color=RED, weight="bold", va="top")
        fig.text(x + 0.064, 0.16, f"“{off[j]}”", fontsize=6.8, color=INK, va="top")
        fig.text(x, 0.065, "$v$, 17 tiles: ", fontsize=6.8, color=RED, weight="bold", va="top")
        fig.text(x + 0.064, 0.065, f"“{on[j]}”", fontsize=6.8, color=INK, va="top")
    fig.savefig(OUT + "teaser.pdf"); fig.savefig(OUT + "teaser.png", dpi=150)

# ---------------- Tiles: schematic + per-view damage (full width) ----------------
def tiles():
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig = plt.figure(figsize=(6.9, 2.0))
    axA = fig.add_axes([0.0, 0.02, 0.56, 0.96]); axA.set_xlim(0, 100); axA.set_ylim(0, 40); axA.axis("off")
    axB = fig.add_axes([0.685, 0.22, 0.29, 0.6])
    def box(x, y, w, h, title, sub, hl=False):
        axA.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                     fc="#e8f1fc" if hl else "white", ec=BLUE if hl else LINE, lw=1.1 if hl else 0.7))
        axA.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=7.6, weight="bold", color=INK)
        axA.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.4, color=INK2, linespacing=1.1)
    def arrow(x0, y0, x1, y1):
        axA.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.7, mutation_scale=7))
    box(1, 15, 17, 10, "Image + $v$", "512 px, 8-bit")
    box(25, 24, 33, 13, "16 local tiles", "upscaled ×4, then cut;\n$v$ stretched ×4")
    for i in range(4):
        for j in range(4):
            axA.add_patch(Rectangle((53.6 + i * 1.1, 32.4 - j * 1.1), 1.0, 1.0, fc="#dcdad3", ec="none"))
    box(25, 3, 33, 13, "1 global tile", "downscaled back to 512 px;\n$v$ at its native scale", hl=True)
    box(65, 15, 16, 10, "SigLIP-B/16", "17 passes")
    box(86, 15, 13, 10, "SmolLM2", "135M")
    arrow(18.6, 21, 24.4, 30); arrow(18.6, 19, 24.4, 10)
    arrow(58.6, 30, 64.4, 22); arrow(58.6, 10, 64.4, 18); arrow(81.6, 20, 85.4, 20)
    axA.text(0.5, 39.2, "(a) SmolVLM's default preprocessing (image splitting on)", fontsize=7.6, color=INK, va="top")
    rows = ["Single tile\n(splitting off)", "Global tile", "16 local tiles"]
    v = [0.184, 0.184, 0.727]; v_rng = [None, None, (0.687, 0.751)]
    r = [0.864, 0.864, 0.770]; r_rng = [None, None, None]
    ys = [2, 1, 0]
    for y, a, b, ar, br in zip(ys, v, r, v_rng, r_rng):
        axB.plot([a, b], [y, y], color=LINE, lw=1.0, zorder=1)
        if ar: axB.plot(ar, [y, y], color=BLUE, lw=2.2, alpha=0.45, solid_capstyle="round", zorder=2)
        if br: axB.plot(br, [y, y], color=GRAY, lw=2.2, alpha=0.45, solid_capstyle="round", zorder=2)
        axB.scatter([b], [y], s=26, color=GRAY, edgecolor="white", linewidth=0.8, zorder=3)
        axB.scatter([a], [y], s=26, color=BLUE, edgecolor="white", linewidth=0.8, zorder=4)
        if ar:
            axB.text(ar[0] - 0.02, y, f"{a:.2f}", ha="right", va="center", fontsize=6.4, color=INK)
            axB.text((br[1] if br else b) + 0.03, y, f"{b:.2f}", ha="left", va="center", fontsize=6.4, color=INK2)
        else:
            axB.text(a, y + 0.24, f"{a:.2f}", ha="center", fontsize=6.4, color=INK)
            axB.text(b, y + 0.24, f"{b:.2f}", ha="center", fontsize=6.4, color=INK2)
    axB.set_yticks(ys); axB.set_yticklabels(rows, fontsize=7); axB.set_ylim(-0.6, 2.7); axB.set_xlim(0, 1.0)
    axB.set_xlabel("Visual-token cosine to clean (1 = unchanged)", fontsize=7)
    axB.grid(axis="x", color="#ecebe6", lw=0.5); axB.set_axisbelow(True)
    for s in ["top", "right", "left"]: axB.spines[s].set_visible(False)
    axB.tick_params(axis="y", length=0); axB.tick_params(axis="x", labelsize=6.5)
    axB.scatter([], [], s=22, color=BLUE, label="universal $v$"); axB.scatter([], [], s=22, color=GRAY, label="random $\\pm\\epsilon$")
    axB.legend(loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=2, frameon=False, fontsize=6.8, handletextpad=0.2)
    fig.text(0.585, 0.965, "(b) Damage per view", fontsize=7.6, color=INK, va="top")
    fig.savefig(OUT + "tiles.pdf"); fig.savefig(OUT + "tiles.png", dpi=150)

# ---------------- Budget sweep + visibility (column width) ----------------
def budget():
    eps = [4, 8, 10, 16]; v_k = [43, 125, 143, 183]; r = [1.0, 3.5, 4.0, 4.0]; n = 200
    fig = plt.figure(figsize=(3.3, 2.55))
    ax = fig.add_axes([0.14, 0.53, 0.83, 0.40])
    lo, hi = zip(*[wilson(k, n) for k in v_k]); vp = [100 * k / n for k in v_k]
    ax.fill_between(eps, lo, hi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(eps, vp, color=BLUE, lw=2, marker="o", ms=4.5, mec="white", mew=0.8)
    ax.plot(eps, r, color=GRAY, lw=2, marker="o", ms=4.5, mec="white", mew=0.8)
    for e, y in zip(eps, vp): ax.text(e, y + 6, f"{y:.1f}", ha="center", fontsize=6.4, color=INK)
    ax.text(16.3, vp[-1], "universal $v$", fontsize=6.8, color=INK, va="center")
    ax.text(16.3, 4, "random $\\pm\\epsilon$", fontsize=6.8, color=INK2, va="center")
    ax.set_xlim(3, 21.5); ax.set_ylim(0, 105); ax.set_xticks(eps); ax.set_xticklabels([str(e) for e in eps], fontsize=6.5); ax.set_xlabel("$\\epsilon$ (in units of 1/255)", fontsize=7, labelpad=1)
    ax.set_ylabel("Captions changed (%)", fontsize=7); ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", color="#ecebe6", lw=0.5); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.set_title("(a) Attack strength vs. budget $\\epsilon$ (real pipeline, 200 images)", fontsize=7.2, color=INK, loc="left", pad=4)
    src = Image.open("/home/claude/pdfimg/img-011.png").convert("RGB")
    cols = [(10, 306), (327, 623), (644, 940), (961, 1257), (1278, 1574)]
    labels = ["clean", "4/255", "8/255", "10/255", "16/255"]
    for k, (c0, c1) in enumerate(cols):
        a = fig.add_axes([0.02 + k * 0.196, 0.02, 0.18, 0.25])
        a.imshow(src.crop((c0, 50, c1 + 1, 347))); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
        a.set_title(labels[k], fontsize=6.6, color=INK2, pad=1.5)
    fig.text(0.02, 0.355, "(b) The flattest 160×160 region among 30 images, zoomed", fontsize=7.2, color=INK)
    fig.savefig(OUT + "budget.pdf"); fig.savefig(OUT + "budget.png", dpi=150)

# ---------------- Defenses: fixed -> naive -> strong against own baseline (column width) ----------------
def defenses():
    D = [  # name, baseline, fixed, naive, strong, strong CI, naive_dagger
        ("JPEG, q75", 71.5, 8.5, 23.0, None, None, False),
        ("Blur, $\\sigma$=1", 71.5, 5.0, 29.5, None, None, False),
        ("Random resize-pad", 71.5, 4.0, 21.0, 29.0, (23.2, 35.6), False),
        ("FT-fixed", 71.5, 4.0, 41.0, None, None, True),
        ("FT-UAT", 71.5, 2.5, 37.5, None, None, True),
        ("Global tile: resize-pad", 60.5, 1.5, 13.0, 35.5, (29.2, 42.3), False),
        ("Global tile: blanked", 59.0, 20.0, 80.0, None, None, False),
    ]
    fig = plt.figure(figsize=(3.3, 2.45)); ax = fig.add_axes([0.33, 0.17, 0.64, 0.72])
    for i, (nm, base, fx, nv, st, ci, dag) in enumerate(D):
        y = len(D) - 1 - i
        last = st if st is not None else nv
        ax.plot([fx, last], [y, y], color=LINE, lw=1.0, zorder=1)
        ax.plot([base, base], [y - 0.32, y + 0.32], color=INK, lw=1.3, zorder=2)
        ax.scatter([fx], [y], s=22, color=GRAY, edgecolor="white", lw=0.7, zorder=3)
        ax.scatter([nv], [y], s=22, color=BLUE_L, edgecolor="white", lw=0.7, zorder=3, marker="D" if dag else "o")
        if st is not None:
            ax.plot(ci, [y, y], color=BLUE, lw=2.4, alpha=0.3, solid_capstyle="round", zorder=2)
            ax.scatter([st], [y], s=26, color=BLUE, edgecolor="white", lw=0.7, zorder=4)
    ax.set_yticks(range(len(D))); ax.set_yticklabels([d[0] for d in D][::-1], fontsize=6.8)
    ax.set_xlim(0, 85); ax.set_ylim(-0.7, len(D) - 0.3)
    ax.set_xlabel("Captions changed (%)", fontsize=7); ax.tick_params(axis="x", labelsize=6.5); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ecebe6", lw=0.5); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    h = [ax.scatter([], [], s=20, color=GRAY), ax.scatter([], [], s=20, color=BLUE_L), ax.scatter([], [], s=22, color=BLUE),
         plt.Line2D([], [], color=INK, lw=1.3, marker="|", ms=7, ls="none")]
    ax.legend(h, ["fixed $v$", "naive adaptive", "strong adaptive", "no defense"], ncol=4, loc="lower left",
              bbox_to_anchor=(-0.48, 1.0), frameon=False, fontsize=6.4, handletextpad=0.15, columnspacing=0.8)
    fig.savefig(OUT + "defenses.pdf"); fig.savefig(OUT + "defenses.png", dpi=150)

if __name__ == "__main__":
    teaser(); tiles(); budget(); defenses()
