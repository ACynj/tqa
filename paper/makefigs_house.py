"""All 5 paper figures in figures4papers house style (ChenLiu-1996).
Palette/typography per references/api.md; sized for IEEE columns."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700", "teal": "#42949E",
    "violet": "#9A4D8E", "ink": "#2B2B2B",
}
P = "/mnt/ssd/tqa/paper/"

def apply_publication_style(base=9.0, lw=1.3):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": base,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": lw, "pdf.fonttype": 42, "figure.dpi": 300,
        "xtick.major.width": lw, "ytick.major.width": lw,
        "legend.frameon": False, "svg.fonttype": "none"})

def finalize(fig, name, pad=0.04):
    import os
    os.makedirs(P + "figs_final", exist_ok=True)
    fig.savefig(f"{P}figs_final/{name}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(f"{P}figs_final/{name}.png", dpi=300, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)

def box(ax, x, y, w, h, text="", fc="white", ec=PALETTE["ink"], fs=8,
        tc=None, lw=1.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                color=tc or PALETTE["ink"], weight="bold" if bold else "normal",
                linespacing=1.3)

def arrow(ax, x1, y1, x2, y2, c=PALETTE["ink"], lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color=c, lw=lw, shrinkA=3, shrinkB=3))

# ============ Fig 1: concept, two-column (7.16 x 3.0) ============
apply_publication_style(base=8.5, lw=1.2)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax = axes[0]
ax.text(-0.2, 10.3, "(a)", fontsize=11, weight="bold")
ax.text(5, 10.0, "Fixed score-factored operators (existing)", ha="center",
        fontsize=9.5, color=PALETTE["red_strong"], weight="bold")
box(ax, 0.3, 7.7, 3.6, 1.15, "query $q_1$\n$(a{,}r_a)\\cap(a{,}r_1)$", fc=PALETTE["red_1"], ec=PALETTE["red_2"], fs=8)
box(ax, 6.1, 7.7, 3.6, 1.15, "query $q_2$\n$(a{,}r_a)\\cap(a{,}r_2)$", fc=PALETTE["red_1"], ec=PALETTE["red_2"], fs=8)
strip = np.array([1.0, 0.2, 0.7, 0.15, 0.9, 0.25, 0.6])
for x in (0.8, 6.6):
    ax.imshow(strip.reshape(1, -1), cmap="Greys", vmin=0, vmax=1.6, aspect="auto",
              extent=(x, x + 2.6, 6.0, 6.6))
    ax.add_patch(plt.Rectangle((x, 6.0), 2.6, 0.6, fill=False, ec=PALETTE["ink"], lw=1.0))
ax.text(2.1, 5.5, "identical marginals", fontsize=7.5, ha="center", style="italic", color="#666")
ax.text(7.9, 5.5, "identical marginals", fontsize=7.5, ha="center", style="italic", color="#666")
box(ax, 2.3, 3.3, 5.4, 1.0, "pointwise $\\prod$ / $\\oplus$ / $(1{-}s)$   (no parameters)",
    fc=PALETTE["red_1"], ec=PALETTE["red_2"], fs=8)
arrow(ax, 2.1, 5.95, 3.8, 4.4, c=PALETTE["red_strong"]); arrow(ax, 7.9, 5.95, 6.2, 4.4, c=PALETTE["red_strong"])
box(ax, 1.7, 1.2, 6.6, 1.25, "identical rankings on $q_1 \\neq q_2$\nstructural collision",
    fc=PALETTE["red_1"], ec=PALETTE["red_strong"], fs=8.5, tc=PALETTE["red_strong"], bold=True, lw=1.8)
arrow(ax, 5.0, 3.3, 5.0, 2.55, c=PALETTE["red_strong"])
ax.text(5.0, 0.55, "operators never see relational context", fontsize=8, ha="center",
        color=PALETTE["red_strong"], style="italic")

ax = axes[1]
ax.text(-0.2, 10.3, "(b)", fontsize=11, weight="bold")
ax.text(5, 10.0, "Witness-gated algebra (ours, learnable)", ha="center",
        fontsize=9.5, color=PALETTE["blue_main"], weight="bold")
box(ax, 0.3, 7.7, 3.6, 1.15, "query $q_1$", fc="#EAF1F8", ec=PALETTE["blue_secondary"], fs=8)
box(ax, 6.1, 7.7, 3.6, 1.15, "query $q_2$", fc="#EAF1F8", ec=PALETTE["blue_secondary"], fs=8)
for x in (0.8, 6.6):
    ax.imshow(strip.reshape(1, -1), cmap="Greys", vmin=0, vmax=1.6, aspect="auto",
              extent=(x, x + 2.6, 6.0, 6.6))
    ax.add_patch(plt.Rectangle((x, 6.0), 2.6, 0.6, fill=False, ec=PALETTE["ink"], lw=1.0))
box(ax, 0.3, 4.45, 3.6, 1.05, "witness evidence\n$(s_i^{\\max},\\,\\bar w_i,\\,\\rho_i)$",
    fc=PALETTE["green_1"], ec=PALETTE["green_3"], fs=7.5)
box(ax, 6.1, 4.45, 3.6, 1.05, "witness evidence\n$(s_i^{\\max},\\,\\bar w_i,\\,\\rho_i)$",
    fc=PALETTE["green_1"], ec=PALETTE["green_3"], fs=7.5)
arrow(ax, 2.1, 5.95, 2.1, 5.6, c=PALETTE["green_3"], lw=1.8)
arrow(ax, 7.9, 5.95, 7.9, 5.6, c=PALETTE["green_3"], lw=1.8)
ax.add_patch(FancyBboxPatch((1.3, 2.15), 7.4, 1.65,
             boxstyle="round,pad=0.02,rounding_size=0.08",
             fc="#DCE8F5", ec=PALETTE["blue_main"], lw=2.0))
ax.text(5.0, 3.35, "gated Aczél–Alsina $t$-norm   (learnable)", ha="center",
        fontsize=9, color=PALETTE["blue_main"], weight="bold")
ax.text(5.0, 2.65, "$T=\\exp[-(\\sum_i(-\\ln s_i)^{\\theta_i})^{1/\\theta_1}]$",
        ha="center", fontsize=8.5, color=PALETTE["blue_main"])
arrow(ax, 2.1, 4.45, 3.5, 3.9, c=PALETTE["blue_main"]); arrow(ax, 7.9, 4.45, 6.5, 3.9, c=PALETTE["blue_main"])
box(ax, 0.9, 0.95, 3.5, 1.05, "ranking of $q_1$\nanswers separated", fc="white",
    ec=PALETTE["teal"], fs=8, tc=PALETTE["teal"], bold=True)
box(ax, 5.6, 0.95, 3.5, 1.05, "ranking of $q_2$\nanswers separated", fc="white",
    ec=PALETTE["teal"], fs=8, tc=PALETTE["teal"], bold=True)
arrow(ax, 3.5, 2.15, 2.6, 2.05, c=PALETTE["teal"]); arrow(ax, 6.5, 2.15, 7.4, 2.05, c=PALETTE["teal"])
ax.text(5.0, 0.3, "$\\theta\\equiv 1$ recovers the fixed operator exactly (Lemma 1)",
        fontsize=7.5, ha="center", color="#666", style="italic")
fig.tight_layout(w_pad=2.5)
finalize(fig, "fig_concept")

# ============ Fig 2: architecture (3.5 x 3.0) ============
apply_publication_style(base=8.0, lw=1.2)
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
box(ax, 1.3, 8.6, 7.4, 1.05, "query plan   $A\\!\\rightarrow\\!P\\!\\rightarrow\\!\\cdots\\!\\rightarrow\\!I\\,/\\,U\\,/\\,N$",
    fc="#F2F2F2", ec=PALETTE["neutral"], fs=8)
ax.add_patch(FancyBboxPatch((0.4, 6.25), 9.2, 1.6,
             boxstyle="round,pad=0.02,rounding_size=0.08",
             fc="#EAF1F8", ec=PALETTE["blue_secondary"], lw=1.6))
ax.text(5.0, 7.38, "vocabulary-free projection $P_r$  (shared)", ha="center",
        fontsize=8.5, color=PALETTE["blue_main"], weight="bold")
ax.text(5.0, 6.72, "relation-graph GNN + conditional propagation (multi-source)",
        ha="center", fontsize=7, color=PALETTE["blue_main"])
arrow(ax, 5.0, 8.6, 5.0, 7.9)
box(ax, 0.4, 4.55, 4.2, 1.0, "branch scores $s_i$", fc="white", fs=8)
box(ax, 5.4, 4.55, 4.2, 1.0, "witness evidence $(\\bar w_i,\\rho_i)$",
    fc=PALETTE["green_1"], ec=PALETTE["green_3"], fs=7.5)
arrow(ax, 3.0, 6.25, 2.5, 5.6); arrow(ax, 7.0, 6.25, 7.5, 5.6)
ax.add_patch(FancyBboxPatch((0.9, 2.05), 8.2, 1.7,
             boxstyle="round,pad=0.02,rounding_size=0.08",
             fc="#DCE8F5", ec=PALETTE["blue_main"], lw=2.0))
ax.text(5.0, 3.28, "gated logical operators (ours)", ha="center",
        fontsize=8.8, color=PALETTE["blue_main"], weight="bold")
ax.text(5.0, 2.60, "$\\theta_i=0.5+3.5\\,\\sigma(\\,g(s_i^{\\max},\\bar w_i,\\rho_i)\\,)$",
        ha="center", fontsize=8, color=PALETTE["blue_main"])
arrow(ax, 2.5, 4.55, 3.6, 3.8, c=PALETTE["blue_main"]); arrow(ax, 7.5, 4.55, 6.4, 3.8, c=PALETTE["blue_main"])
box(ax, 2.2, 0.45, 5.6, 1.0, "entity rankings (zero-shot)", fc="white",
    ec=PALETTE["teal"], fs=8.5, tc=PALETTE["teal"], bold=True)
arrow(ax, 5.0, 2.05, 5.0, 1.5, c=PALETTE["teal"])
ax.text(9.35, 2.9, "fixed ops\n$=\\,\\theta{\\equiv}1$\nspecial case",
        fontsize=6.5, ha="center", color="#888", style="italic", linespacing=1.5)
fig.tight_layout()
finalize(fig, "fig_arch")

# ============ Fig 3: head-to-head grouped bars (7.16 x 2.6, two-col) ============
apply_publication_style(base=8.5, lw=1.1)
types = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"]
uq = [0.357, 0.088, 0.064, 0.355, 0.489, 0.129, 0.176, 0.127, 0.085]
sq = [0.169, 0.031, 0.043, 0.474, 0.347, 0.410, 0.507, 0.963, 0.382]
x = np.arange(len(types)); w = 0.37
fig, ax = plt.subplots(figsize=(7.16, 2.4))
b1 = ax.bar(x - w/2, uq, w, label="UltraQuery", color=PALETTE["red_2"], ec="white", lw=0.6)
b2 = ax.bar(x + w/2, sq, w, label="SQA (ours)", color=PALETTE["blue_main"], ec="white", lw=0.6)
for rect, v in zip(b1, uq): rect.set_alpha(0.95)
ax.axhline(0.208, color=PALETTE["red_strong"], ls=":", lw=1.0)
ax.axhline(0.369, color=PALETTE["blue_main"], ls=":", lw=1.0)
ax.text(8.6, 0.219, "UQ avg", fontsize=6.5, color=PALETTE["red_strong"], ha="right")
ax.text(8.6, 0.380, "SQA avg", fontsize=6.5, color=PALETTE["blue_main"], ha="right")
ax.set_xticks(x); ax.set_xticklabels(types, fontsize=8.5)
ax.tick_params(axis="y", labelsize=8)
ax.set_ylabel("MRR", fontsize=9)
ax.legend(fontsize=8, loc="upper left", handlelength=1.1, borderaxespad=0.1)
ax.set_ylim(0, 1.04)
finalize(fig, "fig_headtohead")

# ============ Fig 4: boundary (3.5 x 2.5) ============
apply_publication_style(base=8.0, lw=1.0)
graphs = ["FB15k-237\n(237 rel)", "WN18RR\n(11 rel)", "CoDEx-M\n(51 rel)"]
fixed = [0.121, 0.124, 0.194]; gated = [0.125, 0.091, 0.141]
gated2 = [0.147, 0.159, 0.187]; free = [0.202, 0.091, 0.141]
x = np.arange(3); w = 0.19
fig, ax = plt.subplots(figsize=(3.5, 2.4))
series = [(fixed, "fixed (order-0)", PALETTE["neutral"]),
          (gated, "gated (order-1)", PALETTE["red_2"]),
          (gated2, "gated (order-2)", PALETTE["blue_main"]),
          (free, "free-form", PALETTE["violet"])]
for k, (vals, lab, c) in enumerate(series):
    ax.bar(x + (k - 1.5) * w, vals, w, label=lab, color=c, ec="white", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(graphs, fontsize=7.5)
ax.tick_params(axis="y", labelsize=7.5)
ax.set_ylabel("zero-shot MRR", fontsize=8.5)
ax.legend(fontsize=6.2, ncol=2, loc="upper left", columnspacing=0.8, handlelength=1.0)
ax.set_ylim(0, 0.26)
ax.annotate("order-2 repairs\nthe boundary", xy=(1.12, 0.162), fontsize=6.5,
            color=PALETTE["blue_main"], xytext=(1.55, 0.205),
            arrowprops=dict(arrowstyle="->", lw=0.9, color=PALETTE["blue_main"]))
finalize(fig, "fig_boundary")

# ============ Fig 5: theta histogram (3.5 x 2.0) ============
import pickle, torch, sys
sys.path.insert(0, "/mnt/ssd/tqa/src")
import tqa.eval_betae as EB
from tqa.model import TQA
apply_publication_style(base=8.0, lw=1.0)
dev = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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
fig, ax = plt.subplots(figsize=(3.5, 1.95))
ax.hist(thetas, bins=25, color=PALETTE["blue_main"], ec="white", lw=0.5, alpha=0.95)
ax.axvline(1.0, color=PALETTE["red_strong"], lw=1.2, ls="--")
ymax = ax.get_ylim()[1]
ax.text(1.006, ymax * 0.88, "$\\theta=1$ (product)", fontsize=7, color=PALETTE["red_strong"])
mu = float(np.mean(thetas))
ax.axvline(mu, color=PALETTE["teal"], lw=1.2)
ax.text(mu - 0.007, ymax * 0.62, f"learned mean {mu:.2f}", fontsize=7,
        color=PALETTE["teal"], ha="right")
ax.set_xlabel("learned gate exponent $\\theta$", fontsize=8.5)
ax.set_ylabel("branches", fontsize=8.5)
ax.tick_params(labelsize=7.5)
finalize(fig, "fig_theta")
print("house-style figures done; theta n =", len(thetas), "mean", round(mu, 3))
