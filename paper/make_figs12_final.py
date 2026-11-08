"""Fig.1 (concept) & Fig.2 (architecture) in the exact visual language of
LMR Fig.2 / ARE Fig.2: rounded stage panels with title bars, node-link KG
minis, segmented vector strips, score bars with %, block arrows, formula
under sub-figures, one yellow-emphasis panel for the learnable core."""
import sys
sys.path.insert(0, "/mnt/ssd/tqa/paper")
from figlib import PALETTE, FigureStyle, apply_publication_style, finalize_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

apply_publication_style(FigureStyle(font_size=7.5, axes_linewidth=0.9))
OUT = "/mnt/ssd/tqa/paper/figs_final"
INK, GRAY = "#272727", "#767676"
BLUES = ["#9DC3E6", "#6FA8DC", "#3D85C6", "#0F4D92"]
GREENS = ["#A9DFBF", "#7DCEA0", "#4E9A62", "#1E8449"]
REDS = ["#F5B7B1", "#EC7063", "#CB4335", "#922B21"]
REL = {"ra": "#0F4D92", "r1": "#7D3C98", "r2": "#CA6F1E"}

def panel(ax, x, y, w, h, title=None, fc="#FFFFFF", ec="#8C8C8C", ts=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.04,rounding_size=0.10",
                 fc=fc, ec=ec, lw=0.9, zorder=1))
    if title:
        ax.add_patch(FancyBboxPatch((x + 0.12, y + h - 0.52), w - 0.24, 0.44,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     fc="#EDF1F5", ec="none", zorder=2))
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center",
                fontsize=ts, weight="bold", zorder=3)

def node(ax, x, y, c, r=0.16, lab=None, fs=13, lc=INK):
    ax.add_patch(Circle((x, y), r, fc=c, ec="white", lw=1.2, zorder=6))
    if lab:
        ax.text(x, y - r - 0.14, lab, ha="center", va="top", fontsize=fs,
                style="italic", color=lc, zorder=6)

def edge(ax, x1, y1, x2, y2, c, lw=1.1, arr=True):
    if arr:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=7, color=c, lw=lw, shrinkA=9, shrinkB=9,
                     zorder=4))
    else:
        ax.plot([x1, x2], [y1, y2], color=c, lw=lw, zorder=3,
                solid_capstyle="round")

def block_arrow(ax, x1, y1, x2, y2, c="#606060"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="simple,head_width=0.55,head_length=0.75,tail_width=0.32",
                 fc="white", ec=c, lw=1.0, mutation_scale=7, zorder=5))

def vstrip(ax, x, y, w, h, colors, label=None, fs=13):
    n = len(colors)
    for k, col in enumerate(colors):
        ax.add_patch(Rectangle((x + k * w / n, y), w / n * 0.9, h,
                     fc=col, ec="white", lw=0.8, zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=INK, lw=0.9, zorder=6))
    if label:
        ax.text(x + w / 2, y - 0.22, label, ha="center", va="top",
                fontsize=fs, style="italic", zorder=6)

def score_bars(ax, x, y, vals, labels, hl=0, w=1.5, gap=0.5, fs=12):
    for k, (v, lab) in enumerate(zip(vals, labels)):
        yy = y - k * gap
        col = "#E67E22" if k == hl else "#BDC3C7"
        ax.add_patch(Rectangle((x, yy - 0.14), w * v, 0.3, fc=col, ec="none", zorder=5))
        ax.text(x + w * v + 0.12, yy, f"{int(v*100)}%", fontsize=fs,
                va="center", zorder=6)
        ax.text(x - 0.15, yy, lab, fontsize=fs, va="center", ha="right", zorder=6)

# ================= Fig 2: architecture (LMR Fig.2 style) =================
fig, ax = plt.subplots(figsize=(7.16, 3.0))
ax.set_xlim(0.1, 23.9); ax.set_ylim(0.3, 9.2); ax.axis("off")

# ---- Input panel ----
panel(ax, 0.2, 0.6, 3.6, 8.2, "Input", ts=16)
node(ax, 1.2, 6.9, INK, 0.17, "$a$", fs=13)
node(ax, 2.9, 7.5, GRAY, 0.14, "$v$", fs=12)
node(ax, 2.9, 6.3, GRAY, 0.14, "$v$", fs=12)
edge(ax, 1.2, 6.9, 2.9, 7.5, REL["ra"], 2.4)
edge(ax, 1.2, 6.9, 2.9, 6.3, REL["r1"], 2.4)
ax.text(2.05, 7.85, "$r_a$", fontsize=5.8, color=REL["ra"], style="italic", ha="center")
ax.text(1.95, 6.35, "$r_1$", fontsize=5.8, color=REL["r1"], style="italic", ha="center")
node(ax, 4.4 - 1.5, 4.6, "white", 0.17, fs=12)
ax.text(2.9, 4.6, "$\\cap$", fontsize=7.2, ha="center", va="center", zorder=7)
edge(ax, 3.02, 7.4, 1.55, 4.75, INK, 1.4)
edge(ax, 3.02, 6.4, 1.55, 4.75, INK, 1.4)
node(ax, 2.9, 2.9, GRAY, 0.14, "$?$", fs=13)
edge(ax, 1.55, 4.45, 2.75, 3.05, INK, 1.4)
ax.text(2.0, 1.9, "query plan\n$A{\\to}P{\\to}\\cdots{\\to}I\ /\ U\ /\ N$",
        fontsize=5.8, ha="center", style="italic", color=GRAY)
# KG mini
np.random.seed(3)
kgx = np.array([[0.8, 3.4], [1.7, 3.75], [2.6, 3.3], [1.5, 2.95], [2.4, 2.7]])
kge = [(0, 1, "ra"), (1, 2, "r1"), (0, 3, "r1"), (3, 4, "ra"), (2, 4, "ra")]
for i, j, rk in kge:
    edge(ax, *kgx[i], *kgx[j], REL[rk], 1.6)
for x, y in kgx:
    node(ax, x + 0.15, y, GRAY, 0.10)
ax.text(2.0, 1.15, "knowledge graph $G$", fontsize=5.5, ha="center",
        style="italic", color=GRAY)

# ---- Stage I: projection ----
panel(ax, 4.4, 0.6, 7.0, 8.2, "Stage I: Vocabulary-free Projection", ts=12.5)
# entity graph mini
ax.text(6.1, 7.35, "entity graph $A_V$", fontsize=6.0, ha="center",
        style="italic", color=GRAY)
np.random.seed(8)
eg = np.array([[5.0, 6.45], [6.0, 6.85], [7.0, 6.35], [5.5, 5.6], [6.7, 5.7]])
esc = np.array([0.95, 0.7, 0.5, 0.35, 0.2])
eedges = [(0, 1, "ra"), (1, 2, "r1"), (0, 3, "r1"), (3, 4, "ra"), (2, 4, "ra"), (1, 4, "r1")]
for i, j, rk in eedges:
    edge(ax, *eg[i], *eg[j], REL[rk], 1.6)
for (x, y), v in zip(eg, esc):
    ax.add_patch(Circle((x, y), 0.12 + 0.05 * v, fc=PALETTE["blue_main"],
                 alpha=0.3 + 0.65 * v, ec="white", lw=1.0, zorder=6))
# relation graph mini
ax.text(9.5, 7.35, "relation graph $A_R$", fontsize=6.0, ha="center",
        style="italic", color=GRAY)
rg = {"ra": (8.8, 6.25), "r1": (10.0, 6.6), "r2": (9.5, 5.55)}
for k, (x, y) in rg.items():
    ax.add_patch(Circle((x, y), 0.08, fc=REL[k], ec="white", lw=1.2, zorder=6))
    ax.text(x, y, f"${k.replace('r','r_')}$", fontsize=5.3, color="white",
            ha="center", va="center", zorder=7)
for a, b in [("ra", "r1"), ("ra", "r2"), ("r1", "r2")]:
    ax.plot([rg[a][0], rg[b][0]], [rg[a][1], rg[b][1]], color=GRAY, lw=1.2, zorder=3)
ax.text(9.5, 4.95, "query-conditioned $\\mathcal{R}$", fontsize=5.5,
        ha="center", style="italic", color=GRAY)
# outputs of stage I: scores + witness strips
vstrip(ax, 5.0, 3.0, 2.6, 0.5, BLUES, "$s_i$ (branch scores)", fs=12)
vstrip(ax, 8.5, 3.0, 2.6, 0.5, GREENS, "$\\bar w_i$ (witness)", fs=12)
ax.text(7.9, 1.55, "conditional propagation (multi-source)", fontsize=5.5,
        ha="center", style="italic", color=GRAY)
ax.text(7.9, 1.0, "$P_r$: GNN over $A_R$ + NBFNet hops", fontsize=5.8,
        ha="center", style="italic")

# ---- Stage II: gated operators (yellow emphasis = the learnable core) ----
panel(ax, 12.2, 0.6, 6.6, 8.2, "Stage II: Gated Logical Operators (ours)", ts=12.5, fc="#FFF9E0")
vstrip(ax, 13.0, 6.75, 2.3, 0.5, BLUES, "$s_i$", fs=12)
vstrip(ax, 13.0, 5.45, 2.3, 0.5, GREENS, "$\\bar w_i$", fs=12)
ax.add_patch(Circle((16.6, 6.9), 0.2, fc="#7D3C98", ec="white", lw=1.2, zorder=6))
ax.text(16.6, 6.9, "$\\rho$", fontsize=5.8, color="white", ha="center", va="center", zorder=7)
ax.text(16.6, 6.45, "signature", fontsize=5.3, ha="center", style="italic", color=GRAY)
gate_y = 4.4
ax.add_patch(FancyBboxPatch((12.55, gate_y - 0.62), 5.5, 1.2,
             boxstyle="round,pad=0.03,rounding_size=0.10", fc="white",
             ec="#B64342", lw=0.8, zorder=4))
ax.text(15.5, gate_y + 0.14, r"$\theta_i = 0.5+3.5 \sigma(g(s_i^{\max},\bar w_i,\rho_i))\in(0.5,4)$",
        fontsize=6.5, ha="center", va="center", zorder=5)
ax.text(15.3, gate_y - 0.30, r"$T_\Theta=\exp[-(\sum_i(-\ln s_i)^{\theta_i})^{1/\theta_1}]$",
        fontsize=5.8, ha="center", va="center", zorder=5, color=PALETTE["blue_main"])
edge(ax, 14.2, 5.4, 14.7, 5.05, GRAY, 1.4)
edge(ax, 14.2, 4.2, 14.7, 3.95, GRAY, 1.4)
ax.text(15.5, 2.62, r"$\theta\equiv 1$ recovers the fixed product exactly",
        fontsize=5.8, ha="center", style="italic", color="#B64342")
ax.text(15.5, 3.28, "learnable: ~15k params, no identifiers",
        fontsize=5.5, ha="center", style="italic", color=GRAY)
# theta curves mini
m = ax.inset_axes([13.6 / 24, 0.85 / 9.6, 3.6 / 24, 1.5 / 9.6])
xs = np.linspace(1e-3, 1 - 1e-3, 80)
for th, col, lw_ in [(0.5, "#C9A9CB", 1.6), (1.0, "#7D3C98", 2.4), (2.0, "#5B2C6F", 1.6), (4.0, "#38174D", 1.6)]:
    T = np.exp(-(((-np.log(xs))**th) + ((-np.log(1 - xs))**th))**(1 / th))
    m.plot(xs, T, color=col, lw=lw_)
m.set_xticks([]); m.set_yticks([])
for s_ in m.spines.values(): s_.set_linewidth(0.8)
ax.text(16.55, 1.55, "Aczél–Alsina family", fontsize=5.5, ha="center",
        style="italic", color="#7D3C98")

# ---- Output panel ----
panel(ax, 19.6, 0.6, 4.2, 8.2, "Scoring", ts=16)
score_bars(ax, 21.0, 7.0, [0.9, 0.55, 0.35, 0.15],
           ["$t_1$", "$t_2$", "$t_3$", "$t_4$"], hl=0, w=1.9, gap=0.62, fs=12.5)
ax.text(21.7, 4.35, r"$p(t | \mathrm{plan},G)$", fontsize=6.2, ha="center", style="italic")
node(ax, 20.6, 3.2, "#E67E22", 0.17, "$t_1$", fs=13)
node(ax, 22.0, 2.7, "#BDC3C7", 0.15, "$t_2$", fs=13)
node(ax, 23.4, 2.2, "#BDC3C7", 0.15, "$t_3$", fs=13)
edge(ax, 20.75, 3.05, 21.85, 2.8, GRAY, 1.2, arr=False)
edge(ax, 22.15, 2.55, 23.25, 2.3, GRAY, 1.2, arr=False)
ax.text(21.6, 1.35, "zero-shot on any graph", fontsize=6.0, ha="center",
        style="italic", color=PALETTE["teal"])

# ---- block arrows ----
block_arrow(ax, 3.9, 4.7, 4.35, 4.7)
block_arrow(ax, 11.5, 4.7, 12.15, 4.7)
block_arrow(ax, 18.9, 4.7, 19.55, 4.7)
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
finalize_figure(fig, f"{OUT}/fig_arch", formats=["pdf", "png"], dpi=300, pad=0.02)

# ================= Fig 1: concept (ARE Fig.2 style stacked panels) ========
fig, axes = plt.subplots(2, 1, figsize=(14, 7.6))
for ax in axes:
    ax.set_xlim(0.2, 23.8); ax.set_ylim(0.2, 9.4); ax.axis("off")

def kg_block(ax, x0, y0):
    """dashed panel with a KG whose r1/r2 edges share successors."""
    ax.add_patch(FancyBboxPatch((x0, y0), 8.2, 7.6,
                 boxstyle="round,pad=0.05,rounding_size=0.12", fc="white",
                 ec="#9AA5B1", lw=0.8, ls=(0, (5, 3))))
    ax_ = x0 + 1.4
    node(ax, ax_, y0 + 4.6, INK, 0.17, "$a$", fs=13)
    for ty, lab in [(y0 + 6.3, "$t_1$"), (y0 + 5.5, "$t_2$")]:
        node(ax, x0 + 4.6, ty, GRAY, 0.14, lab)
        edge(ax, ax_, y0 + 4.6, x0 + 4.6, ty, REL["ra"], 2.4)
    for ty, lab in [(y0 + 3.6, "$t_3$"), (y0 + 2.8, "$t_4$")]:
        node(ax, x0 + 4.6, ty, GRAY, 0.14, lab)
    edge(ax, ax_, y0 + 4.6, x0 + 4.45, y0 + 3.72, REL["r1"], 2.2)
    edge(ax, ax_, y0 + 4.6, x0 + 4.45, y0 + 2.88, REL["r1"], 2.2)
    edge(ax, ax_, y0 + 4.6, x0 + 4.72, y0 + 3.52, REL["r2"], 2.2)
    edge(ax, ax_, y0 + 4.6, x0 + 4.72, y0 + 2.68, REL["r2"], 2.2)
    ax.text(x0 + 2.8, y0 + 5.6, "$r_a$", fontsize=5.8, color=REL["ra"],
            style="italic", ha="center")
    ax.text(x0 + 2.7, y0 + 3.75, "$r_1$", fontsize=5.8, color=REL["r1"],
            style="italic", ha="center")
    ax.text(x0 + 3.0, y0 + 2.9, "$r_2$", fontsize=5.8, color=REL["r2"],
            style="italic", ha="center")
    # relation-graph inset
    ix, iy = x0 + 5.8, y0 + 6.4
    ax.add_patch(FancyBboxPatch((ix - 0.25, iy - 0.8), 2.3, 1.55,
                 boxstyle="round,pad=0.03,rounding_size=0.08", fc="#FBFBFB",
                 ec="#B0B8C1", lw=1.0))
    for k, (dx, dy) in {"ra": (0.15, 0.35), "r1": (1.5, 0.4), "r2": (0.85, -0.35)}.items():
        ax.add_patch(Circle((ix + dx, iy + dy), 0.08, fc=REL[k], ec="white", lw=1.0, zorder=6))
        ax.text(ix + dx, iy + dy, f"${k.replace('r','r_')}$", fontsize=4.8,
                color="white", ha="center", va="center", zorder=7)
    for (dx1, dy1, dx2, dy2) in [(0.3, 0.3, 1.35, 0.36), (0.3, 0.25, 0.8, -0.24),
                                  (0.95, -0.24, 1.42, 0.3)]:
        ax.plot([ix + dx1, ix + dx2], [iy + dy1, iy + dy2], color=GRAY, lw=1.0, zorder=3)
    ax.text(x0 + 4.1, y0 + 1.5, "$\\mathrm{Succ}(r_1)=\\mathrm{Succ}(r_2)$",
            fontsize=6.0, ha="center", style="italic", color=GRAY)

# ---- (a) fixed ----
ax = axes[0]
kg_block(ax, 0.3, 1.0)
ax.text(0.2, 9.15, "(a)  Fixed score-factored operators (existing)", fontsize=7.7,
        weight="bold", color="#B64342", va="top")
# right column: identical strips -> product -> identical ranking
vstrip(ax, 10.5, 6.9, 3.0, 0.55, ["#B9B9B9"] * 4 + ["#6E6E6E"] + ["#B9B9B9"] * 3,
       "$s(a{,}r_1)$", fs=12.5)
vstrip(ax, 15.5, 6.9, 3.0, 0.55, ["#B9B9B9"] * 4 + ["#6E6E6E"] + ["#B9B9B9"] * 3,
       "$s(a{,}r_2)$", fs=12.5)
ax.text(14.5, 5.75, r"pointwise $\prod\ /\ \oplus\ /\ (1{-}s)$   (no parameters)",
        fontsize=6.5, ha="center",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=2))
block_arrow(ax, 14.5, 5.45, 14.5, 4.75)
vstrip(ax, 11.3, 3.55, 3.0, 0.55, ["#B9B9B9"] * 4 + ["#6E6E6E"] + ["#B9B9B9"] * 3,
       "ranking of $q_1$", fs=12.5)
vstrip(ax, 16.3, 3.55, 3.0, 0.55, ["#B9B9B9"] * 4 + ["#6E6E6E"] + ["#B9B9B9"] * 3,
       "ranking of $q_2$", fs=12.5)
for qx in (12.8, 17.8):
    ax.add_patch(FancyBboxPatch((qx - 0.62, 4.55), 1.24, 0.6,
                 boxstyle="round,pad=0.02,rounding_size=0.15",
                 fc="#F4F6F7", ec=GRAY, lw=0.7))
    ax.text(qx, 4.85, "$q_1$" if qx < 15 else "$q_2$", fontsize=6.2,
            ha="center", va="center")
ax.text(14.5, 2.55, "identical rankings on $q_1\\neq q_2$  (collision)",
        fontsize=7.2, ha="center", color="#B64342", weight="bold")
ax.text(14.5, 1.7, "operators never see which relation produced a score",
        fontsize=6.0, ha="center", style="italic", color=GRAY)

# ---- (b) ours ----
ax = axes[1]
kg_block(ax, 0.3, 1.0)
ax.text(0.2, 9.15, "(b)  Witness-gated algebra (ours, learnable)", fontsize=7.7,
        weight="bold", color=PALETTE["blue_main"], va="top")
# right column: evidence -> gate -> separated rankings
ax.add_patch(FancyBboxPatch((10.3, 5.5), 8.6, 1.9,
             boxstyle="round,pad=0.04,rounding_size=0.1", fc="#F0FAF0",
             ec=PALETTE["green_3"], lw=0.8))
ax.text(14.6, 7.05, "witness evidence", fontsize=6.5, ha="center",
        color="#1E8449", weight="bold")
vstrip(ax, 11.1, 5.85, 2.4, 0.45, GREENS, "$\\bar w$", fs=12)
ax.add_patch(Circle((15.1, 6.35), 0.08, fc="#7D3C98", ec="white", lw=1.2, zorder=6))
ax.text(15.1, 6.35, "$\\rho$", fontsize=5.8, color="white", ha="center", va="center", zorder=7)
ax.text(16.6, 6.35, "distinct\nsignatures", fontsize=5.3, ha="left",
        va="center", style="italic", color="#7D3C98")
ax.add_patch(FancyArrowPatch((8.8, 7.4), (10.6, 6.6), arrowstyle="-|>",
             mutation_scale=7, color=PALETTE["green_3"], lw=1.0,
             connectionstyle="arc3,rad=-0.2", ls=(0, (4, 3))))
ax.add_patch(FancyBboxPatch((10.3, 3.4), 8.6, 1.55,
             boxstyle="round,pad=0.04,rounding_size=0.1", fc="#FFF9E0",
             ec=PALETTE["blue_main"], lw=0.9))
ax.text(14.65, 4.5, r"$\theta_i = g(s_i^{\max},\bar w_i,\rho_i)$,  $T_\Theta$ gates the family",
        fontsize=6.2, ha="center", color=PALETTE["blue_main"])
ax.text(14.65, 3.85, r"$\theta\equiv 1\Rightarrow$ exact fixed operator (Lemma 1)",
        fontsize=5.8, ha="center", style="italic", color="#B64342")
block_arrow(ax, 14.6, 5.4, 14.6, 5.05)
block_arrow(ax, 14.6, 3.3, 14.6, 2.95)
vstrip(ax, 11.0, 1.95, 3.0, 0.55, ["#1E8449"] + ["#D5E8DD"] * 3 + ["#A9C9B5"] + ["#D5E8DD"] * 3,
       "$q_1\\!:\\ t_1$ top", fs=12.5)
vstrip(ax, 16.0, 1.95, 3.0, 0.55, ["#D5E8DD"] + ["#1E8449"] + ["#A9C9B5"] * 2 + ["#D5E8DD"] * 4,
       "$q_2\\!:\\ t_2$ top", fs=12.5)
ax.text(14.5, 1.05, "separated rankings $\\Rightarrow$ both queries answered",
        fontsize=7.2, ha="center", color=PALETTE["teal"], weight="bold")
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, hspace=0.08)
finalize_figure(fig, f"{OUT}/fig_concept", formats=["pdf", "png"], dpi=300, pad=0.02)
print("final LMR/ARE-recipe figures done")
