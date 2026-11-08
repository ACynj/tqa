"""Fig.1 (concept) & Fig.2 (architecture) in LMR/ARE house style:
node-link KG diagrams, colored edges = relations, dashed panels, pill chips,
no flowchart boxes. Flat vector, serif-ish labels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "pdf.fonttype": 42, "figure.dpi": 300})
INK = "#2B2B2B"
# semantic palette (ARE/LMR style): edge color = relation identity
C = {"ra": "#0F4D92",   # r_a  blue
     "r1": "#7D3C98",   # r_1  purple
     "r2": "#CA6F1E",   # r_2  orange
     "node": "#85929E", "anchor": "#1B2631",
     "good": "#1E8449", "bad": "#B64342", "hi": "#B03A2E"}
P = "/mnt/ssd/tqa/paper/"

def panel(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc="white", ec="#9AA5B1", lw=1.1, ls=(0, (4, 3))))

def node(ax, x, y, c, r=0.14, label=None, lc=None, fs=7, lw=0):
    ax.add_patch(Circle((x, y), r, fc=c, ec="white", lw=lw or 0.8, zorder=5))
    if label:
        ax.text(x, y - r - 0.16, label, ha="center", va="top", fontsize=fs,
                style="italic", color=lc or INK)

def edge(ax, x1, y1, x2, y2, c, lw=1.6, ls="-", ah=0):
    if ah:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, color=c, lw=lw, ls=ls,
                                     shrinkA=8, shrinkB=8, zorder=3))
    else:
        ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2,
                solid_capstyle="round")

def pill(ax, x, y, w, h, text, fc, ec, fs=7, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or INK)

def rel_label(ax, x, y, t, c, fs=7):
    ax.text(x, y, t, fontsize=fs, color=c, style="italic", ha="center")

# ==================== Fig 1: concept (7.16 x 3.0) ====================
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.05))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# ---------- (a) existing ----------
ax = axes[0]
panel(ax, 0.1, 0.1, 9.8, 8.6)
ax.text(0.35, 9.6, "(a)", fontsize=10, weight="bold")
ax.text(5.0, 9.55, "Fixed score-factored operators (existing)", ha="center",
        fontsize=8.8, weight="bold", color=C["bad"])
# KG: anchor a; targets t1..t4; r_a edges to t1,t2; r1 & r2 edges BOTH to t3,t4 (same successors!)
ax_ = 1.3
node(ax, ax_, 5.0, C["anchor"], 0.16, "$a$")
for ty, lab in [(6.7, "$t_1$"), (5.9, "$t_2$")]:
    node(ax, 4.2, ty, C["node"], 0.13, lab)
    edge(ax, ax_, 5.0, 4.2, ty, C["ra"], 1.8)
rel_label(ax, 2.6, 6.6, "$r_a$", C["ra"]); rel_label(ax, 2.6, 5.35, "$r_a$", C["ra"])
# r1 and r2 to SAME targets t3,t4 (offset slightly) => identical marginals
for ty, lab in [(4.6, "$t_3$"), (3.8, "$t_4$")]:
    node(ax, 4.2, ty, C["node"], 0.13, lab)
edge(ax, ax_, 5.0, 4.1, 4.72, C["r1"], 1.6)   # r1
edge(ax, ax_, 5.0, 4.1, 3.88, C["r1"], 1.6)
edge(ax, ax_, 5.0, 4.32, 4.5, C["r2"], 1.6)   # r2 (offset, same targets)
edge(ax, ax_, 5.0, 4.32, 3.66, C["r2"], 1.6)
rel_label(ax, 2.55, 4.35, "$r_1$", C["r1"]); rel_label(ax, 2.75, 3.45, "$r_2$", C["r2"])
ax.text(2.7, 2.9, "same successors\n$\\Rightarrow$ identical marginals",
        fontsize=6.8, ha="center", style="italic", color="#666")
# score strips (identical) then product then collision
strip = np.array([1, .2, .7, .15])
for x0 in (5.0, 6.6):
    ax.imshow(strip.reshape(1, -1), cmap="Greys", vmin=0, vmax=1.5, aspect="auto",
              extent=(x0, x0 + 1.5, 6.15, 6.7))
    ax.add_patch(plt.Rectangle((x0, 6.15), 1.5, 0.55, fill=False, ec=INK, lw=0.7))
ax.text(6.55, 6.95, "identical marginals", fontsize=6.5, ha="center", style="italic", color="#666")
ax.text(5.75, 5.35, "$\\times$", fontsize=13, ha="center", color=INK)
ax.add_patch(FancyArrowPatch((5.75, 6.1), (5.75, 5.75), arrowstyle="-|>",
                             mutation_scale=11, color=INK, lw=1.2))
ax.add_patch(FancyArrowPatch((5.75, 5.15), (5.75, 4.6), arrowstyle="-|>",
                             mutation_scale=11, color=INK, lw=1.2))
ax.text(5.75, 4.9, r"pointwise $\prod$, no parameters", fontsize=7, ha="center")
# two queries -> identical ranking
for qx, qlab in ((4.4, "$q_1$"), (7.1, "$q_2$")):
    pill(ax, qx - 0.55, 3.35, 1.1, 0.62, qlab, "#FDEDEC", C["bad"])
ax.text(5.75, 3.0, r"$\Downarrow$", fontsize=11, ha="center", color=C["bad"])
for x0 in (4.5, 7.0):
    ax.imshow(strip.reshape(1, -1), cmap="Greys", vmin=0, vmax=1.5, aspect="auto",
              extent=(x0, x0 + 1.5, 1.7, 2.25))
    ax.add_patch(plt.Rectangle((x0, 1.7), 1.5, 0.55, fill=False, ec=C["bad"], lw=1.0))
ax.text(5.75, 1.15, "identical rankings on $q_1\\neq q_2$  (collision)",
        fontsize=7.5, ha="center", color=C["bad"], weight="bold")
ax.text(5.75, 0.55, "operators never see which relation produced a score",
        fontsize=6.8, ha="center", style="italic", color="#777")

# ---------- (b) ours ----------
ax = axes[1]
panel(ax, 0.1, 0.1, 9.8, 8.6)
ax.text(0.35, 9.6, "(b)", fontsize=10, weight="bold")
ax.text(5.0, 9.55, "Witness-gated algebra (ours, learnable)", ha="center",
        fontsize=8.8, weight="bold", color=C["ra"])
# same KG skeleton
ax_ = 1.3
node(ax, ax_, 5.0, C["anchor"], 0.16, "$a$")
for ty, lab in [(6.7, "$t_1$"), (5.9, "$t_2$")]:
    node(ax, 3.6, ty, C["node"], 0.13, lab)
    edge(ax, ax_, 5.0, 3.6, ty, C["ra"], 1.8)
for ty, lab in [(4.6, "$t_3$"), (3.8, "$t_4$")]:
    node(ax, 3.6, ty, C["node"], 0.13, lab)
edge(ax, ax_, 5.0, 3.5, 4.72, C["r1"], 1.6)
edge(ax, ax_, 5.0, 3.5, 3.88, C["r1"], 1.6)
edge(ax, ax_, 5.0, 3.72, 4.5, C["r2"], 1.6)
edge(ax, ax_, 5.0, 3.72, 3.66, C["r2"], 1.6)
rel_label(ax, 2.35, 4.35, "$r_1$", C["r1"]); rel_label(ax, 2.55, 3.45, "$r_2$", C["r2"])
# relation-graph inset: r_a, r1, r2 nodes with DISTINCT signatures (different gray-edge patterns)
ix, iy = 4.35, 7.6
ax.add_patch(FancyBboxPatch((ix - 0.15, iy - 0.75), 2.5, 1.5,
             boxstyle="round,pad=0.04,rounding_size=0.1", fc="#FBFBFB", ec="#B0B8C1", lw=0.8))
node(ax, ix + 0.35, iy + 0.35, C["ra"], 0.11); ax.text(ix + 0.35, iy + 0.35, "$r_a$", fontsize=5.5, color="white", ha="center", va="center", zorder=6)
node(ax, ix + 1.55, iy + 0.4, C["r1"], 0.11); ax.text(ix + 1.55, iy + 0.4, "$r_1$", fontsize=5.5, color="white", ha="center", va="center", zorder=6)
node(ax, ix + 0.95, iy - 0.4, C["r2"], 0.11); ax.text(ix + 0.95, iy - 0.4, "$r_2$", fontsize=5.5, color="white", ha="center", va="center", zorder=6)
# distinct interaction patterns (gray thin edges, different counts)
for (x1, y1, x2, y2) in [(ix+0.45, iy+0.3, ix+1.45, iy+0.38),
                          (ix+0.42, iy+0.25, ix+0.9, iy-0.32),
                          (ix+1.05, iy-0.32, ix+1.5, iy+0.32)]:
    ax.plot([x1, x2], [y1, y2], color="#7F8C8D", lw=0.7, zorder=1)
ax.text(ix + 1.1, iy + 0.85, "relation graph: distinct", fontsize=6.2, ha="center", style="italic", color="#555")
ax.text(ix + 1.1, iy - 0.95, "signatures $\\rho_1 \\neq \\rho_2$", fontsize=6.2, ha="center", style="italic", color="#555")
# arrow inset -> gate
ax.add_patch(FancyArrowPatch((ix + 1.2, iy - 1.35), (5.4, 5.6), arrowstyle="-|>",
                             mutation_scale=11, color=C["good"], lw=1.4,
                             connectionstyle="arc3,rad=-0.25", ls=(0, (3, 2))))
# score strips differ now
strip2 = strip.copy(); strip2[2] = 0.15
for x0, s_ in ((4.75, strip), (6.35, strip2)):
    ax.imshow(s_.reshape(1, -1), cmap="Greys", vmin=0, vmax=1.5, aspect="auto",
              extent=(x0, x0 + 1.4, 5.05, 5.6))
    ax.add_patch(plt.Rectangle((x0, 5.05), 1.4, 0.55, fill=False, ec=INK, lw=0.7))
# theta curves inset (Aczel-Alsala family)
tx, ty = 8.05, 6.4
xs = np.linspace(1e-3, 1, 60)
ax_mini = ax.inset_axes([tx / 10, ty / 10 * 0.86, 1.55 / 10 * 0.98, 1.45 / 10 * 0.98])
for th, col in [(0.5, "#9FB8CF"), (1.0, C["ra"]), (2.0, "#6E93B8"), (4.0, "#39618A")]:
    T = np.exp(-(((-np.log(xs))**th) + ((-np.log(1 - xs + 1e-9))**th))**(1 / th))
    ax_mini.plot(xs, T, color=col, lw=1.1)
ax_mini.text(0.32, 0.75, "$\\theta$", fontsize=7, color=C["ra"])
ax_mini.set_xticks([]); ax_mini.set_yticks([])
for s in ax_mini.spines.values(): s.set_linewidth(0.6)
ax.text(tx + 0.75, ty - 1.7, "gated $t$-norm family", fontsize=6.4, ha="center", color=C["ra"])
# gate flow
ax.add_patch(FancyArrowPatch((5.45, 5.0), (5.45, 4.35), arrowstyle="-|>",
                             mutation_scale=12, color=C["ra"], lw=1.6))
ax.text(5.45, 4.62, r"$T_\Theta$: $\theta_i = g(s_i^{\max},\bar w_i,\rho_i)$",
        fontsize=7.5, ha="center", color=C["ra"])
# separated rankings (different strips) -> different top
s3 = np.array([1, .2, .15, .1]); s4 = np.array([.15, 1, .3, .1])
for x0, s_ in ((4.3, s3), (6.6, s4)):
    ax.imshow(s_.reshape(1, -1), cmap="Greys", vmin=0, vmax=1.5, aspect="auto",
              extent=(x0, x0 + 1.4, 2.35, 2.9))
    ax.add_patch(plt.Rectangle((x0, 2.35), 1.4, 0.55, fill=False, ec=C["good"], lw=1.0))
ax.text(5.0, 2.62, "$q_1\\!: t_1$", fontsize=7, ha="center", color=C["good"], weight="bold")
ax.text(7.3, 2.62, "$q_2\\!: t_2$", fontsize=7, ha="center", color=C["good"], weight="bold")
ax.text(5.75, 1.75, "separated rankings $\\Rightarrow$ both answered",
        fontsize=7.5, ha="center", color=C["good"], weight="bold")
ax.text(5.75, 1.15, "theta=1 recovers the product exactly (Lemma 1)",
        fontsize=6.8, ha="center", style="italic", color="#777")
fig.tight_layout(w_pad=2.0)
fig.savefig(P + "figs_final/fig_concept.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(P + "figs_final/fig_concept.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ==================== Fig 2: architecture (3.5 x 2.9) ====================
fig, ax = plt.subplots(figsize=(3.5, 2.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
# --- query plan as node-link DAG (top) ---
node(ax, 1.1, 8.6, C["anchor"], 0.15, "$a$", fs=6.5)
node(ax, 3.4, 9.1, C["node"], 0.12, "", fs=6)
node(ax, 3.4, 8.1, C["node"], 0.12, "", fs=6)
edge(ax, 1.1, 8.6, 3.4, 9.1, C["ra"], 1.5, ah=1)
edge(ax, 1.1, 8.6, 3.4, 8.1, C["r1"], 1.5, ah=1)
rel_label(ax, 2.2, 9.15, "$r_a$", C["ra"], 6); rel_label(ax, 2.2, 8.15, "$r_1$", C["r1"], 6)
node(ax, 5.1, 8.6, "white", 0.16, "", fs=6, lw=1.2)
ax.text(5.1, 8.6, "$\\cap$", fontsize=8, ha="center", va="center", zorder=6)
edge(ax, 3.4, 9.1, 5.1, 8.65, INK, 1.0, ah=1)
edge(ax, 3.4, 8.1, 5.1, 8.55, INK, 1.0, ah=1)
node(ax, 6.6, 8.6, C["node"], 0.12, "$?$", fs=6.5)
edge(ax, 5.25, 8.6, 6.45, 8.6, INK, 1.0, ah=1)
ax.text(3.9, 9.8, "query plan", fontsize=7.5, weight="bold", color=INK)
# --- projection on KG subgraph (middle, dashed panel) ---
panel(ax, 0.3, 3.9, 6.4, 4.1)
ax.text(0.55, 7.6, "vocabulary-free projection $P_r$", fontsize=7.5, weight="bold", color=C["ra"])
ax.text(0.55, 7.15, "relation-graph GNN + conditional propagation", fontsize=6, color="#666")
# KG mini graph with score propagation shading
np.random.seed(5)
kg_nodes = np.array([[1.3, 5.9], [2.4, 6.5], [3.3, 5.8], [2.0, 5.0], [3.0, 4.7],
                     [4.2, 5.4], [1.6, 4.3], [4.6, 6.3], [5.4, 5.0]])
kg_edges = [(0, 1, "ra"), (1, 2, "r1"), (0, 3, "r1"), (3, 4, "ra"), (2, 4, "ra"),
            (4, 5, "r1"), (3, 6, "ra"), (5, 8, "ra"), (1, 7, "r1"), (7, 8, "ra")]
score = np.array([0.95, 0.8, 0.55, 0.6, 0.45, 0.3, 0.2, 0.35, 0.15])
for i, j, rk in kg_edges:
    edge(ax, *kg_nodes[i], *kg_nodes[j], C[rk], 1.1)
for (x, y), s_ in zip(kg_nodes, score):
    ax.add_patch(Circle((x, y), 0.13 + 0.05 * s_, fc=C["ra"], alpha=0.25 + 0.7 * s_,
                         ec="white", lw=0.6, zorder=5))
ax.text(3.2, 4.15, "multi-source scores propagate", fontsize=6, ha="center",
        style="italic", color="#666")
# relation graph inset (right of panel)
ix, iy = 6.0, 5.6  # unused; keep inside panel bounds instead
# --- witness evidence (green dashed box, bottom-right of panel) ---
ax.add_patch(FancyBboxPatch((4.35, 6.55), 2.15, 1.1,
             boxstyle="round,pad=0.03,rounding_size=0.1", fc="#F0FAF0",
             ec="#8BCF8B", lw=1.0, ls=(0, (3, 2))))
ax.text(5.42, 7.42, "witness evidence", fontsize=6.2, ha="center", color="#1E8449", weight="bold")
ax.text(5.42, 6.95, r"$(\bar w_i,\ \rho_i)$", fontsize=7, ha="center", color="#1E8449")
# --- gated operators (bottom panel) ---
ax.add_patch(FancyBboxPatch((0.3, 1.6), 6.4, 1.85,
             boxstyle="round,pad=0.04,rounding_size=0.1", fc="#F3F8FD",
             ec=C["ra"], lw=1.4))
ax.text(3.5, 3.05, "gated logical operators (ours)", fontsize=7.8, ha="center",
        weight="bold", color=C["ra"])
ax.text(3.5, 2.45, "$\\theta_i = 0.5 + 3.5\\,\\sigma(g(s_i^{\\max},\\bar w_i,\\rho_i))$",
        fontsize=7, ha="center", color=C["ra"])
ax.text(3.5, 1.9, "fixed operators $=\\ \\theta\\!\\equiv\\!1$ special case",
        fontsize=6.2, ha="center", style="italic", color="#777")
# arrows down
ax.add_patch(FancyArrowPatch((3.0, 3.9), (3.0, 3.5), arrowstyle="-|>",
                             mutation_scale=12, color=C["ra"], lw=1.4))
ax.add_patch(FancyArrowPatch((5.4, 6.55), (5.4, 3.5), arrowstyle="-|>",
                             mutation_scale=12, color="#1E8449", lw=1.3,
                             connectionstyle="arc3,rad=0.2"))
# --- ranked candidates output (node row sorted by color intensity) ---
ax.text(8.35, 7.6, "ranked answers", fontsize=7.3, weight="bold", color="#42949E", ha="center")
vals = [1.0, 0.75, 0.5, 0.3]
for k, v_ in enumerate(vals):
    ax.add_patch(Circle((7.7 + 0.0, 6.7 - k * 0.62), 0.15, fc=C["ra"], alpha=0.2 + 0.75 * v_,
                         ec="white", lw=0.8, zorder=5))
    ax.add_patch(Circle((8.9, 6.7 - k * 0.62), 0.1, fc="#BDC3C7", ec="white", lw=0.6, zorder=4))
    ax.plot([7.85, 8.8], [6.7 - k * 0.62, 6.7 - k * 0.62], color="#B0B8C1", lw=0.8, zorder=1)
ax.text(8.3, 4.05, "zero-shot on any graph", fontsize=6.2, ha="center",
        style="italic", color="#42949E")
ax.add_patch(FancyArrowPatch((6.7, 2.6), (7.45, 5.2), arrowstyle="-|>",
                             mutation_scale=12, color="#42949E", lw=1.4))
fig.tight_layout()
fig.savefig(P + "figs_final/fig_arch.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(P + "figs_final/fig_arch.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("LMR/ARE-style concept+arch done")
