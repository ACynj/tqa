"""Polished publication figures v2 (no overlap, concept figure highlighted)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42,
                     "figure.dpi": 300, "axes.linewidth": 0.6})
GRAY, BLUE, RED, PURPLE, AMBER = "#7f8c8d", "#2e6da4", "#c0392b", "#7d3c98", "#d68910"
P = "/mnt/ssd/tqa/paper/"

def box(ax, x, y, w, h, text, fc="white", ec="black", fs=7.5, lw=0.8, style="round,pad=0.02", tc="black", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, mutation_scale=1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight)

def arrow(ax, x1, y1, x2, y2, color="black", lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=8, color=color, lw=lw, shrinkA=1, shrinkB=1))

# ============ Fig 1: concept (problem vs innovation), two-column ============
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# ---- Panel A: existing ----
ax = axes[0]
ax.set_title("(a) Existing: fixed score-factored operators", fontsize=8.5, color=GRAY, pad=4)
box(ax, 0.3, 8.6, 3.2, 1.0, "query $q_1=(a,r_a){\\cap}(a,r_1)$", fc="#f2f2f2", ec=GRAY, fs=6.8)
box(ax, 6.2, 8.6, 3.5, 1.0, "query $q_2=(a,r_a){\\cap}(a,r_2)$", fc="#f2f2f2", ec=GRAY, fs=6.8)
# identical score strips
for x0, lab in [(1.0, "branch scores"), ]:
    pass
np.random.seed(3)
strip = np.array([1, 0.15, 0.7, 0.1, 0.9, 0.2, 0.55, 0.12])
for (x, y) in [(1.3, 6.4), (2.2, 4.9)]:
    ax.imshow(strip.reshape(1, -1), cmap="Greys", vmin=0, vmax=1.4, aspect="auto",
              extent=(x, x + 2.6, y, y + 0.55))
    ax.add_patch(plt.Rectangle((x, y), 2.6, 0.55, fill=False, ec=GRAY, lw=0.7))
for (x, y) in [(7.0, 6.4), (7.9, 4.9)]:
    ax.imshow(strip.reshape(1, -1), cmap="Greys", vmin=0, vmax=1.4, aspect="auto",
              extent=(x, x + 2.6, y, y + 0.55))
    ax.add_patch(plt.Rectangle((x, y), 2.6, 0.55, fill=False, ec=GRAY, lw=0.7))
ax.text(2.6, 5.75, "identical marginals", fontsize=6.3, color=GRAY, ha="center", style="italic")
ax.text(8.3, 5.75, "identical marginals", fontsize=6.3, color=GRAY, ha="center", style="italic")
# pointwise product box
box(ax, 3.0, 3.2, 4.0, 0.95, r"pointwise $\prod$ / $\oplus$ / $1{-}s$  (no parameters)", fc="#e8e8e8", ec=GRAY, fs=7)
arrow(ax, 2.6, 4.85, 4.2, 4.2, color=GRAY); arrow(ax, 8.2, 4.85, 5.8, 4.2, color=GRAY)
# collision output
box(ax, 2.6, 1.4, 4.8, 1.1, "identical rankings\ncollision: $q_1 \\neq q_2$ answered alike", fc="#fdecea", ec=RED, fs=7, tc=RED)
arrow(ax, 5.0, 3.2, 5.0, 2.55, color=GRAY)
ax.text(5.0, 0.6, "structure-blind: operators never see relation context", fontsize=6.5, ha="center", color=GRAY, style="italic")

# ---- Panel B: ours ----
ax = axes[1]
ax.set_title("(b) Ours: witness-gated logical algebra (learnable)", fontsize=8.5, color=RED, pad=4)
box(ax, 0.3, 8.6, 3.2, 1.0, "query $q_1$", fc="#f2f2f2", ec=GRAY, fs=7)
box(ax, 6.2, 8.6, 3.2, 1.0, "query $q_2$", fc="#f2f2f2", ec=GRAY, fs=7)
strip2 = strip.copy(); strip2[3] = 0.8
for (x, y, s_) in [(1.1, 6.4, strip), (7.0, 6.4, strip2)]:
    ax.imshow(s_.reshape(1, -1), cmap="Greys", vmin=0, vmax=1.4, aspect="auto",
              extent=(x, x + 2.4, y, y + 0.55))
    ax.add_patch(plt.Rectangle((x, y), 2.4, 0.55, fill=False, ec=GRAY, lw=0.7))
# witness evidence badges
box(ax, 0.5, 5.0, 3.6, 0.9, "witness evidence $(s^{\\max},\\bar w,\\rho)$", fc="#fdf2e3", ec=AMBER, fs=6.6, lw=0.9)
box(ax, 5.9, 5.0, 3.6, 0.9, "witness evidence $(s^{\\max},\\bar w,\\rho)$", fc="#fdf2e3", ec=AMBER, fs=6.6, lw=0.9)
arrow(ax, 2.3, 6.35, 2.3, 5.95, color=AMBER); arrow(ax, 8.1, 6.35, 8.1, 5.95, color=AMBER)
# gated t-norm (the innovation, highlighted)
ax.add_patch(FancyBboxPatch((1.6, 2.9), 6.8, 1.35, boxstyle="round,pad=0.06",
                            fc="#fdecea", ec=RED, lw=1.4))
ax.text(5.0, 3.9, "gated Aczel-Alsala t-norm family", ha="center", fontsize=7.4, color=RED, weight="bold")
ax.text(5.0, 3.28, "T = exp[ -( sum_i (-ln s_i)^theta_i )^(1/theta_1) ]", ha="center", fontsize=7, color=RED)
arrow(ax, 2.3, 5.0, 3.6, 4.3, color=RED); arrow(ax, 8.1, 5.0, 6.4, 4.3, color=RED)
# separated outputs
box(ax, 1.4, 1.2, 3.3, 1.0, "ranking of $q_1$\nanswers separated", fc="white", ec=BLUE, fs=6.8, tc=BLUE)
box(ax, 5.3, 1.2, 3.3, 1.0, "ranking of $q_2$\nanswers separated", fc="white", ec=BLUE, fs=6.8, tc=BLUE)
arrow(ax, 3.6, 2.9, 3.0, 2.25, color=BLUE); arrow(ax, 6.4, 2.9, 7.0, 2.25, color=BLUE)
ax.text(5.0, 0.5, r"$\theta\!=\!1$ recovers the product exactly — fixed ops are the embedded special case",
        fontsize=6.4, ha="center", color=RED, style="italic")
fig.tight_layout(w_pad=1.5)
fig.savefig(P + "fig_concept.pdf", bbox_inches="tight"); plt.close(fig)

# ============ Fig 2: architecture (vertical, no overlap) ============
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
box(ax, 2.0, 8.7, 6.0, 1.0, "query plan  $A \\rightarrow P \\rightarrow \\cdots \\rightarrow I\\,/\\,U\\,/\\,N$", fc="#f2f2f2", fs=7.2)
box(ax, 1.2, 6.6, 7.6, 1.1, "vocabulary-free projection $P_r$\nrelation-graph GNN + conditional propagation (multi-source)", fc="#eaf2f8", ec=BLUE, fs=6.8)
arrow(ax, 5.0, 8.7, 5.0, 7.75)
box(ax, 0.4, 4.7, 4.2, 0.95, "branch scores $s_i$", fc="white", fs=7)
box(ax, 5.4, 4.7, 4.2, 0.95, "witness evidence $(\\bar w_i,\\rho_i)$", fc="#fdf2e3", ec=AMBER, fs=6.8)
arrow(ax, 3.5, 6.6, 2.5, 5.7); arrow(ax, 6.5, 6.6, 7.5, 5.7)
ax.add_patch(FancyBboxPatch((1.2, 2.3), 7.6, 1.5, boxstyle="round,pad=0.06", fc="#fdecea", ec=RED, lw=1.3))
ax.text(5.0, 3.35, "gated logical operators  (learnable)", ha="center", fontsize=7.4, color=RED, weight="bold")
ax.text(5.0, 2.72, "gated Aczel-Alsala t-norm family, gate g reads (s-max, w-bar, rho)", ha="center", fontsize=6.2, color=RED)
arrow(ax, 2.5, 4.7, 3.6, 3.85, color=RED); arrow(ax, 7.5, 4.7, 6.4, 3.85, color=RED)
box(ax, 2.6, 0.7, 4.8, 0.9, "entity rankings (zero-shot)", fc="white", fs=7.2)
arrow(ax, 5.0, 2.3, 5.0, 1.65)
ax.text(9.1, 3.05, "fixed ops =\n$\\theta\\equiv1$\nspecial case", fontsize=5.8, color=GRAY, ha="center", style="italic")
fig.tight_layout(); fig.savefig(P + "fig_arch.pdf", bbox_inches="tight"); plt.close(fig)

# ============ Fig 3: head-to-head (no overlap) ============
types = ["1p","2p","3p","2i","3i","ip","pi","2u","up"]
uq = [0.357,0.088,0.064,0.355,0.489,0.129,0.176,0.127,0.085]
sq = [0.169,0.031,0.043,0.474,0.347,0.410,0.507,0.963,0.382]
x = np.arange(len(types)); w = 0.39
fig, ax = plt.subplots(figsize=(3.5, 2.2))
ax.bar(x-w/2, uq, w, label="UltraQuery", color=GRAY, ec="k", lw=0.3)
ax.bar(x+w/2, sq, w, label="SQA (ours)", color=RED, ec="k", lw=0.3)
ax.axhline(0.208, color=GRAY, ls=":", lw=0.7); ax.axhline(0.369, color=RED, ls=":", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(types, fontsize=7)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylabel("MRR", fontsize=8); ax.legend(fontsize=6.5, frameon=False, loc="upper left")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(P + "fig_headtohead.pdf"); plt.close(fig)

# ============ Fig 4: boundary + repair (cleaner) ============
graphs = ["FB15k-237\n(237 rel)", "WN18RR\n(11 rel)", "CoDEx-M\n(51 rel)"]
fixed=[0.121,0.124,0.194]; gated=[0.125,0.091,0.141]; gated2=[0.147,0.159,0.141]; learned=[0.202,0.091,0.141]
x = np.arange(3); w = 0.19
fig, ax = plt.subplots(figsize=(3.5, 2.2))
ax.bar(x-1.5*w, fixed, w, label="fixed (order-0)", color=GRAY, ec="k", lw=0.3)
ax.bar(x-0.5*w, gated, w, label="gated (order-1)", color=RED, ec="k", lw=0.3)
ax.bar(x+0.5*w, gated2, w, label="gated (order-2)", color=PURPLE, ec="k", lw=0.3)
ax.bar(x+1.5*w, learned, w, label="free-form", color=AMBER, ec="k", lw=0.3)
ax.set_xticks(x); ax.set_xticklabels(graphs, fontsize=7)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylabel("zero-shot MRR", fontsize=8)
ax.legend(fontsize=5.8, frameon=False, ncol=2, loc="upper left", columnspacing=0.8, handlelength=1.1)
ax.spines[["top","right"]].set_visible(False)
ax.annotate("order-2 repairs\nthe boundary", xy=(1.13, 0.157), fontsize=6, color=PURPLE,
            xytext=(1.52, 0.175), arrowprops=dict(arrowstyle="->", lw=0.6, color=PURPLE))
fig.tight_layout(); fig.savefig(P + "fig_boundary.pdf"); plt.close(fig)

# ============ Fig 5: theta (polished) ============
import pickle, torch, sys
sys.path.insert(0, "/mnt/ssd/tqa/src")
import tqa.eval_betae as EB
from tqa.model import TQA
dev = torch.device("cuda:0")
m = TQA(d=64, ops="gated").to(dev)
m.load_state_dict(torch.load("/mnt/ssd/tqa/runs/ft5_gated.pt", map_location=dev))
m.eval()
D = "/mnt/ssd/ultra_data/query-datasets/FB15k-237-betae"
q = pickle.load(open(f"{D}/test-queries.pkl", "rb"))
thetas = []
orig = m.ops.intersect
def spy(sl, fl, rf=None):
    for s_, f_ in zip(sl, fl):
        w_ = (s_/(s_.sum(dim=1, keepdim=True)+1e-6)).unsqueeze(-1)
        fsum = (f_*w_).sum(dim=1)
        t = 0.5+3.5*torch.sigmoid(m.ops.gate(torch.cat([s_.max(dim=1,keepdim=True).values, fsum], -1))+m.ops.gate_logit)
        thetas.extend(t.flatten().tolist())
    return orig(sl, fl, rf)
m.ops.intersect = spy
qs = list(q[(('e',('r',)),('e',('r',)))])[:120]
with torch.no_grad():
    for i in range(0, len(qs), 16):
        progs, _ = zip(*[EB.to_program(c) for c in qs[i:i+16]])
        m(EB.g, list(progs), dev)
m.ops.intersect = orig
fig, ax = plt.subplots(figsize=(3.5, 1.9))
ax.hist(thetas, bins=25, color=RED, ec="k", lw=0.3, alpha=0.92)
ax.axvline(1.0, color="k", lw=0.8, ls="--")
ax.text(1.01, ax.get_ylim()[1]*0.88, r"$\theta=1$ (product)", fontsize=6.5)
ax.axvline(float(np.mean(thetas)), color=BLUE, lw=0.8)
ax.text(float(np.mean(thetas))-0.005, ax.get_ylim()[1]*0.65, f"mean {np.mean(thetas):.2f}", fontsize=6.5, color=BLUE, ha="right")
ax.set_xlabel(r"learned gate exponent $\theta$", fontsize=8); ax.set_ylabel("count", fontsize=8)
ax.tick_params(labelsize=7)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(P + "fig_theta.pdf"); plt.close(fig)
print("all figs v2 done; theta n =", len(thetas))
