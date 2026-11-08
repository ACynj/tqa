"""All 5 TQA paper figures via figlib (figures4papers api.md implementation),
drawn in LMR/ARE visual language: node-link KGs, colored edges = relations,
dashed rounded panels, pill query chips. Uses apply_publication_style,
finalize_figure per the skill workflow."""
import sys
sys.path.insert(0, "/mnt/ssd/tqa/paper")
from figlib import (PALETTE, FigureStyle, apply_publication_style,
                    finalize_figure, make_grouped_bar)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

import os
OUT = "/mnt/ssd/tqa/paper/figs_final"
INK, GRAY = "#272727", "#767676"
REL = {"ra": PALETTE["blue_main"], "r1": PALETTE["violet"], "r2": "#CA6F1E"}

apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2.0))

def panel(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.06,rounding_size=0.14",
                 fc="white", ec="#9AA5B1", lw=2.0, ls=(0, (5, 3))))

def node(ax, x, y, c, r=0.15, label=None, fs=15, lc=INK):
    ax.add_patch(Circle((x, y), r, fc=c, ec="white", lw=1.2, zorder=5))
    if label:
        ax.text(x, y - r - 0.18, label, ha="center", va="top", fontsize=fs,
                style="italic", color=lc)

def edge(ax, x1, y1, x2, y2, c, lw=2.4, arr=False, ls="-"):
    if arr:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=16, color=c, lw=lw, ls=ls,
                     shrinkA=10, shrinkB=10, zorder=3))
    else:
        ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2,
                 solid_capstyle="round")

def pill(ax, cx, cy, w, h, text, fc, ec, fs=15, tc=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.18", fc=fc, ec=ec, lw=1.6))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight)

def rlab(ax, x, y, t, c, fs=14):
    ax.text(x, y, t, fontsize=fs, color=c, style="italic", ha="center")

def strip(ax, x, y, w, h, vals, ec=INK):
    n = len(vals)
    for k, v in enumerate(vals):
        ax.add_patch(plt.Rectangle((x + k * w / n, y), w / n * 0.92, h * min(v, 1.0),
                     fc=INK, alpha=0.15 + 0.75 * v, ec="none", zorder=2))
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=ec, lw=1.2, zorder=3))

# ============ Fig 1: concept (two-column) ============
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# ---- (a) existing ----
ax = axes[0]
panel(ax, 0.1, 0.1, 9.8, 8.4)
ax.text(0.25, 9.85, "(a)", fontsize=22, weight="bold", va="top")
ax.text(5.0, 9.25, "Fixed score-factored operators (existing)", ha="center",
        fontsize=18, weight="bold", color=PALETTE["red_strong"], va="top")
node(ax, 1.4, 5.0, INK, 0.17, "$a$", fs=15)
for ty, lab in [(6.8, "$t_1$"), (6.0, "$t_2$")]:
    node(ax, 4.3, ty, GRAY, 0.14, lab)
    edge(ax, 1.4, 5.0, 4.3, ty, REL["ra"], 2.6)
rlab(ax, 2.7, 6.75, "$r_a$", REL["ra"]); rlab(ax, 2.7, 5.4, "$r_a$", REL["ra"])
for ty, lab in [(4.7, "$t_3$"), (3.9, "$t_4$")]:
    node(ax, 4.3, ty, GRAY, 0.14, lab)
edge(ax, 1.4, 5.0, 4.15, 4.82, REL["r1"], 2.4)
edge(ax, 1.4, 5.0, 4.15, 3.98, REL["r1"], 2.4)
edge(ax, 1.4, 5.0, 4.4, 4.62, REL["r2"], 2.4)
edge(ax, 1.4, 5.0, 4.4, 3.78, REL["r2"], 2.4)
rlab(ax, 2.55, 4.45, "$r_1$", REL["r1"]); rlab(ax, 2.8, 3.5, "$r_2$", REL["r2"])
ax.text(2.8, 2.75, "same successors $\\Rightarrow$ identical marginals",
        fontsize=14, ha="center", style="italic", color=GRAY, zorder=7,
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=2.0))
s = [1, .2, .7, .15]
for x0 in (5.1, 6.8):
    strip(ax, x0, 6.2, 1.5, 0.58, s)
ax.text(6.7, 7.15, "identical marginals", fontsize=14, ha="center",
        style="italic", color=GRAY)
ax.add_patch(FancyArrowPatch((6.7, 6.1), (6.7, 5.5), arrowstyle="-|>",
             mutation_scale=16, color=INK, lw=2))
ax.text(6.7, 5.15, r"pointwise $\prod$  (no parameters)", fontsize=15, ha="center",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=2.0), zorder=7)
ax.add_patch(FancyArrowPatch((6.7, 4.8), (6.7, 4.2), arrowstyle="-|>",
             mutation_scale=16, color=INK, lw=2))
for qx in (4.3, 7.7):
    pill(ax, qx, 3.35, 1.25, 0.66, "$q_1$" if qx < 6 else "$q_2$",
         "#F4F6F7", GRAY, fs=15)
ax.text(6.0, 2.35, r"$\Downarrow$", fontsize=20, ha="center", color=PALETTE["red_strong"])
for x0 in (4.6, 7.05):
    strip(ax, x0, 1.35, 1.5, 0.58, s, ec=PALETTE["red_strong"])
ax.text(6.0, 0.72, "identical rankings on $q_1\\neq q_2$  (collision)",
        fontsize=15, ha="center", color=PALETTE["red_strong"], weight="bold")

# ---- (b) ours ----
ax = axes[1]
panel(ax, 0.1, 0.1, 9.8, 8.4)
ax.text(0.25, 9.85, "(b)", fontsize=22, weight="bold", va="top")
ax.text(5.0, 9.25, "Witness-gated algebra (ours, learnable)", ha="center",
        fontsize=18, weight="bold", color=PALETTE["blue_main"], va="top")
node(ax, 1.2, 4.6, INK, 0.17, "$a$", fs=15)
for ty, lab in [(6.5, "$t_1$"), (5.7, "$t_2$")]:
    node(ax, 3.4, ty, GRAY, 0.14, lab)
    edge(ax, 1.2, 4.6, 3.4, ty, REL["ra"], 2.6)
for ty, lab in [(4.4, "$t_3$"), (3.6, "$t_4$")]:
    node(ax, 3.4, ty, GRAY, 0.14, lab)
edge(ax, 1.2, 4.6, 3.25, 4.52, REL["r1"], 2.4)
edge(ax, 1.2, 4.6, 3.25, 3.68, REL["r1"], 2.4)
edge(ax, 1.2, 4.6, 3.5, 4.32, REL["r2"], 2.4)
edge(ax, 1.2, 4.6, 3.5, 3.48, REL["r2"], 2.4)
rlab(ax, 2.2, 4.2, "$r_1$", REL["r1"]); rlab(ax, 2.45, 3.2, "$r_2$", REL["r2"])
# relation-graph inset
ix, iy = 4.5, 8.0
ax.add_patch(FancyBboxPatch((ix - 0.2, iy - 0.85), 2.8, 1.6,
             boxstyle="round,pad=0.05,rounding_size=0.12", fc="#FBFBFB",
             ec="#B0B8C1", lw=1.2))
node(ax, ix + 0.4, iy + 0.4, REL["ra"], 0.12)
ax.text(ix + 0.4, iy + 0.4, "$r_a$", fontsize=11, color="white", ha="center", va="center", zorder=6)
node(ax, ix + 1.8, iy + 0.45, REL["r1"], 0.12)
ax.text(ix + 1.8, iy + 0.45, "$r_1$", fontsize=11, color="white", ha="center", va="center", zorder=6)
node(ax, ix + 1.1, iy - 0.45, REL["r2"], 0.12)
ax.text(ix + 1.1, iy - 0.45, "$r_2$", fontsize=11, color="white", ha="center", va="center", zorder=6)
for (x1, y1, x2, y2) in [(ix+.5, iy+.35, ix+1.7, iy+.42),
                          (ix+.5, iy+.28, ix+1.05, iy-.35),
                          (ix+1.2, iy-.35, ix+1.75, iy+.35)]:
    ax.plot([x1, x2], [y1, y2], color=GRAY, lw=1.2, zorder=1)
ax.text(ix + 1.2, iy + 1.02, "relation graph", fontsize=13, ha="center",
        style="italic", color=GRAY, bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5), zorder=7)
ax.text(ix + 1.2, iy - 1.05, "distinct signatures $\\rho_1\\!\\neq\\!\\rho_2$",
        fontsize=13, ha="center", style="italic", color=GRAY)
ax.add_patch(FancyArrowPatch((ix + 1.3, iy - 1.35), (5.3, 5.9),
             arrowstyle="-|>", mutation_scale=16, color=PALETTE["green_3"],
             lw=2.2, connectionstyle="arc3,rad=-0.25", ls=(0, (4, 3))))
# theta curves inset
tx, ty = 7.55, 6.55
m = ax.inset_axes([tx / 10, ty / 10, 1.5 / 10, 1.4 / 10])
xs = np.linspace(1e-3, 1 - 1e-3, 80)
for th, col, lw_ in [(0.5, "#A9C4DD", 2), (1.0, PALETTE["blue_main"], 3),
                     (2.0, "#5B84AE", 2), (4.0, "#2C5580", 2)]:
    T = np.exp(-(((-np.log(xs))**th) + ((-np.log(1 - xs))**th))**(1 / th))
    m.plot(xs, T, color=col, lw=lw_)
m.text(0.33, 0.72, "$\\theta$", fontsize=15, color=PALETTE["blue_main"])
m.set_xticks([]); m.set_yticks([])
for sp in m.spines.values(): sp.set_linewidth(1.0)
ax.text(tx + 0.7, ty - 1.62, "gated $t$-norm family", fontsize=14,
        ha="center", color=PALETTE["blue_main"],
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5), zorder=7)
# differing strips
s3 = [1, .2, .15, .1]; s4 = [.15, 1, .3, .1]
strip(ax, 4.55, 5.05, 1.4, 0.55, s3)
strip(ax, 6.25, 5.05, 1.4, 0.55, s4)
ax.add_patch(FancyArrowPatch((6.0, 4.9), (6.0, 4.35), arrowstyle="-|>",
             mutation_scale=16, color=PALETTE["blue_main"], lw=2.4))
ax.text(6.0, 4.55, "$T_\\Theta$, gate reads $(s_i^{\\max},\\bar w_i,\\rho_i)$",
        fontsize=15, ha="center", color=PALETTE["blue_main"],
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=2.0), zorder=7)
for qx in (4.3, 7.7):
    pill(ax, qx, 3.35, 1.25, 0.66, "$q_1$" if qx < 6 else "$q_2$",
         "#F4F6F7", GRAY, fs=15)
strip(ax, 4.55, 1.5, 1.4, 0.55, s3, ec=PALETTE["teal"])
strip(ax, 6.4, 1.5, 1.4, 0.55, s4, ec=PALETTE["teal"])
ax.text(5.25, 0.95, "$t_1$ top", fontsize=14, ha="center", color=PALETTE["teal"], weight="bold")
ax.text(7.1, 0.95, "$t_2$ top", fontsize=14, ha="center", color=PALETTE["teal"], weight="bold")
ax.text(6.0, 0.72, "separated rankings $\\Rightarrow$ both answered",
        fontsize=15, ha="center", color=PALETTE["teal"], weight="bold")
fig.tight_layout(w_pad=3, pad=2)
finalize_figure(fig, f"{OUT}/fig_concept", formats=["pdf", "png"], dpi=300)

# ============ Fig 2: architecture (single column) ============
fig, ax = plt.subplots(figsize=(7, 5.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
# query plan DAG
node(ax, 1.1, 8.7, INK, 0.16, "$a$", fs=14)
node(ax, 3.3, 9.15, GRAY, 0.13, fs=13)
node(ax, 3.3, 8.15, GRAY, 0.13, fs=13)
edge(ax, 1.1, 8.7, 3.3, 9.15, REL["ra"], 2.2, arr=True)
edge(ax, 1.1, 8.7, 3.3, 8.15, REL["r1"], 2.2, arr=True)
rlab(ax, 2.1, 9.3, "$r_a$", REL["ra"], 13); rlab(ax, 2.1, 8.2, "$r_1$", REL["r1"], 13)
node(ax, 4.9, 8.65, "white", 0.17, fs=13)
ax.text(4.9, 8.65, "$\\cap$", fontsize=16, ha="center", va="center", zorder=6)
edge(ax, 3.45, 9.12, 4.78, 8.72, INK, 1.6, arr=True)
edge(ax, 3.45, 8.2, 4.78, 8.58, INK, 1.6, arr=True)
node(ax, 6.3, 8.65, GRAY, 0.13, "$?$", fs=14)
edge(ax, 5.06, 8.65, 6.15, 8.65, INK, 1.6, arr=True)
ax.text(3.7, 9.8, "query plan", fontsize=16, weight="bold")
# projection panel
panel(ax, 0.3, 3.9, 6.5, 4.0)
ax.text(0.6, 7.5, "vocabulary-free projection $P_r$", fontsize=16,
        weight="bold", color=PALETTE["blue_main"])
ax.text(0.6, 7.02, "relation-graph GNN + conditional propagation", fontsize=13, color=GRAY)
np.random.seed(7)
kg = np.array([[1.3, 5.9], [2.4, 6.4], [3.3, 5.7], [2.0, 4.9], [3.0, 4.6],
               [4.2, 5.3], [1.6, 4.2], [4.5, 6.2], [5.3, 4.9]])
edges_ = [(0, 1, "ra"), (1, 2, "r1"), (0, 3, "r1"), (3, 4, "ra"), (2, 4, "ra"),
          (4, 5, "r1"), (3, 6, "ra"), (5, 8, "ra"), (1, 7, "r1"), (7, 8, "ra")]
sc = np.array([.95, .8, .55, .6, .45, .3, .2, .35, .15])
for i, j, rk in edges_:
    edge(ax, *kg[i], *kg[j], REL[rk], 1.8)
for (x, y), v in zip(kg, sc):
    ax.add_patch(Circle((x, y), 0.14 + 0.06 * v, fc=PALETTE["blue_main"],
                 alpha=0.25 + 0.7 * v, ec="white", lw=1.0, zorder=5))
ax.text(3.2, 4.1, "multi-source scores propagate", fontsize=13, ha="center",
        style="italic", color=GRAY)
ax.add_patch(FancyBboxPatch((4.5, 6.5), 2.1, 1.05,
             boxstyle="round,pad=0.04,rounding_size=0.12", fc=PALETTE["green_1"],
             ec=PALETTE["green_3"], lw=1.6, ls=(0, (4, 3))))
ax.text(5.55, 7.28, "witness evidence", fontsize=13, ha="center",
        color="#1E8449", weight="bold")
ax.text(5.55, 6.85, "$(\\bar w_i,\\ \\rho_i)$", fontsize=14, ha="center", color="#1E8449")
# gated operators
ax.add_patch(FancyBboxPatch((0.3, 1.5), 6.5, 1.9,
             boxstyle="round,pad=0.05,rounding_size=0.12", fc="#EAF1F8",
             ec=PALETTE["blue_main"], lw=2.2))
ax.text(3.55, 3.05, "gated logical operators (ours)", fontsize=16, ha="center",
        weight="bold", color=PALETTE["blue_main"])
ax.text(3.55, 2.42, "$\\theta_i = 0.5+3.5\\,\\sigma(\\,g(s_i^{\\max},\\bar w_i,\\rho_i)\\,)$",
        fontsize=14, ha="center", color=PALETTE["blue_main"])
ax.text(3.55, 1.82, "fixed operators $=\\ \\theta\\!\\equiv\\!1$ special case",
        fontsize=12, ha="center", style="italic", color=GRAY)
ax.add_patch(FancyArrowPatch((3.0, 3.9), (3.0, 3.48), arrowstyle="-|>",
             mutation_scale=16, color=PALETTE["blue_main"], lw=2.2))
ax.add_patch(FancyArrowPatch((5.5, 6.5), (5.5, 3.48), arrowstyle="-|>",
             mutation_scale=16, color=PALETTE["green_3"], lw=2.0,
             connectionstyle="arc3,rad=0.2"))
# ranked answers
ax.text(8.35, 7.6, "ranked answers", fontsize=15, weight="bold",
        color=PALETTE["teal"], ha="center")
for k, v in enumerate([1.0, 0.72, 0.45, 0.22]):
    yy = 6.6 - k * 0.66
    ax.add_patch(Circle((7.7, yy), 0.16, fc=PALETTE["blue_main"],
                 alpha=0.2 + 0.75 * v, ec="white", lw=1.0, zorder=5))
    ax.add_patch(Circle((8.9, yy), 0.11, fc="#BDC3C7", ec="white", lw=0.8, zorder=4))
    ax.plot([7.88, 8.78], [yy, yy], color="#B0B8C1", lw=1.2, zorder=1)
ax.text(8.3, 3.95, "zero-shot on any graph", fontsize=13, ha="center",
        style="italic", color=PALETTE["teal"])
ax.add_patch(FancyArrowPatch((6.85, 2.5), (7.5, 5.3), arrowstyle="-|>",
             mutation_scale=16, color=PALETTE["teal"], lw=2.2))
fig.tight_layout(pad=2)
finalize_figure(fig, f"{OUT}/fig_arch", formats=["pdf", "png"], dpi=300)
print("skill-style concept+arch done")
