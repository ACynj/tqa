"""Publication-quality figures v3 — top-venue style:
sans-serif, muted professional palette, bold panel labels, no chartjunk."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8, "pdf.fonttype": 42, "figure.dpi": 300,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})
# seaborn "deep"-style muted palette
C = {"gray": "#8C8C8C", "blue": "#4C72B0", "red": "#C44E52",
     "purple": "#8172B2", "amber": "#CCB974", "green": "#55A868",
     "lgray": "#E8E8E8", "lblue": "#DCE6F2", "lred": "#F2DCDB",
     "lamber": "#F5EEDC", "dtext": "#333333"}
P = "/mnt/ssd/tqa/paper/"

def box(ax, x, y, w, h, text="", fc="white", ec="#555555", fs=7, lw=0.9,
        tc=None, bold=False, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                color=tc or C["dtext"], weight="bold" if bold else "normal",
                linespacing=1.35)

def arr(ax, x1, y1, x2, y2, c="#555555", lw=0.9, cs=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, color=c, lw=lw,
                                 connectionstyle=cs or "arc3,rad=0",
                                 shrinkA=2, shrinkB=2))

def panel(ax, x, y, s):
    ax.text(x, y, s, fontsize=10, weight="bold", color="black", ha="left", va="top")

def score_strip(ax, x, y, w, h, vals, ec="#777777"):
    cmap = plt.cm.Grays
    ax.imshow(np.array(vals).reshape(1, -1), cmap=cmap, vmin=0, vmax=1.6,
              aspect="auto", extent=(x, x + w, y, y + h))
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=ec, lw=0.7))

# ================= Fig 1: concept (double column) =================
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# ---- (a) existing ----
ax = axes[0]
panel(ax, -0.1, 10.2, "(a)")
ax.text(5.0, 9.75, "Fixed score-factored operators  (existing)",
        ha="center", fontsize=8.6, color=C["gray"], weight="bold")
box(ax, 0.35, 7.9, 3.5, 1.05, "query  $q_1$\n$(a,r_a)\\cap(a,r_1)$", fc=C["lgray"], ec=C["gray"], fs=7.2)
box(ax, 6.15, 7.9, 3.5, 1.05, "query  $q_2$\n$(a,r_a)\\cap(a,r_2)$", fc=C["lgray"], ec=C["gray"], fs=7.2)
strip = [1.0, 0.2, 0.7, 0.15, 0.9, 0.25, 0.6]
score_strip(ax, 0.75, 6.15, 2.7, 0.55, strip)
score_strip(ax, 3.35, 6.15, 0.0 + 0.5, 0.55, strip[2:3])  # tiny gap piece removed
score_strip(ax, 6.55, 6.15, 2.7, 0.55, strip)
ax.texts[-1].set_text("")
ax.text(2.1, 5.55, "branch marginals", fontsize=6.6, ha="center", color="#666666", style="italic")
ax.text(7.9, 5.55, "branch marginals", fontsize=6.6, ha="center", color="#666666", style="italic")
box(ax, 2.6, 3.55, 4.8, 0.95, "pointwise  $\\prod$ / $\\oplus$ / $(1{-}s)$     (no parameters)",
    fc=C["lgray"], ec=C["gray"], fs=7.2)
arr(ax, 2.1, 6.1, 3.9, 4.6, c=C["gray"]); arr(ax, 7.9, 6.1, 6.1, 4.6, c=C["gray"])
box(ax, 2.0, 1.45, 6.0, 1.15, "identical rankings\n$q_1 \\neq q_2$ answered alike  (collision)",
    fc=C["lred"], ec=C["red"], fs=7.4, tc=C["red"], bold=True, lw=1.2)
arr(ax, 5.0, 3.55, 5.0, 2.68, c=C["red"], lw=1.1)
ax.text(5.0, 0.62, "operators never see relational structure",
        fontsize=7.0, ha="center", color=C["red"], style="italic")

# ---- (b) ours ----
ax = axes[1]
panel(ax, -0.1, 10.2, "(b)")
ax.text(5.0, 9.75, "Witness-gated logical algebra  (ours, learnable)",
        ha="center", fontsize=8.6, color=C["red"], weight="bold")
box(ax, 0.35, 7.9, 3.5, 1.05, "query  $q_1$", fc=C["lgray"], ec=C["gray"], fs=7.2)
box(ax, 6.15, 7.9, 3.5, 1.05, "query  $q_2$", fc=C["lgray"], ec=C["gray"], fs=7.2)
score_strip(ax, 0.75, 6.15, 2.7, 0.55, strip)
score_strip(ax, 6.55, 6.15, 2.7, 0.55, strip)
box(ax, 0.35, 4.65, 3.5, 1.0, "witness evidence\n$(s_i^{\\max},\\ \\bar{w}_i,\\ \\rho_i)$",
    fc=C["lamber"], ec=C["amber"], fs=6.9)
box(ax, 6.15, 4.65, 3.5, 1.0, "witness evidence\n$(s_i^{\\max},\\ \\bar{w}_i,\\ \\rho_i)$",
    fc=C["lamber"], ec=C["amber"], fs=6.9)
arr(ax, 2.1, 6.1, 2.1, 5.75, c=C["amber"], lw=1.1)
arr(ax, 7.9, 6.1, 7.9, 5.75, c=C["amber"], lw=1.1)
ax.add_patch(FancyBboxPatch((1.55, 2.35), 6.9, 1.55,
             boxstyle="round,pad=0.012,rounding_size=0.03",
             fc=C["lred"], ec=C["red"], lw=1.5))
ax.text(5.0, 3.42, "gated Aczél–Alsina $t$-norm   (learnable)", ha="center",
        fontsize=7.6, color=C["red"], weight="bold")
ax.text(5.0, 2.80, "$T=\\exp[-(\\sum_i(-\\ln s_i)^{\\theta_i})^{1/\\theta_1}]$",
        ha="center", fontsize=7.4, color=C["red"])
arr(ax, 2.1, 4.65, 3.6, 3.98, c=C["red"], lw=1.1)
arr(ax, 7.9, 4.65, 6.4, 3.98, c=C["red"], lw=1.1)
box(ax, 1.05, 0.95, 3.4, 1.05, "ranking of $q_1$\nanswers separated",
    fc="white", ec=C["blue"], fs=7.2, tc=C["blue"], bold=True)
box(ax, 5.55, 0.95, 3.4, 1.05, "ranking of $q_2$\nanswers separated",
    fc="white", ec=C["blue"], fs=7.2, tc=C["blue"], bold=True)
arr(ax, 3.6, 2.35, 2.75, 2.05, c=C["blue"], lw=1.0)
arr(ax, 6.4, 2.35, 7.25, 2.05, c=C["blue"], lw=1.0)
ax.text(5.0, 0.28, "$\\theta\\equiv 1$ recovers the fixed operator exactly (Lemma 1)",
        fontsize=6.9, ha="center", color=C["gray"], style="italic")
fig.tight_layout(w_pad=2.2)
fig.savefig(P + "fig_concept.pdf", bbox_inches="tight"); plt.close(fig)

# ================= Fig 2: architecture =================
fig, ax = plt.subplots(figsize=(3.5, 2.75))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
box(ax, 1.5, 8.75, 7.0, 0.95, "query plan   $A\\!\\rightarrow\\!P\\!\\rightarrow\\!\\cdots\\!\\rightarrow\\!I\\,/\\,U\\,/\\,N$",
    fc=C["lgray"], ec=C["gray"], fs=7.0)
ax.add_patch(FancyBboxPatch((0.55, 6.35), 8.9, 1.55,
             boxstyle="round,pad=0.012,rounding_size=0.03",
             fc=C["lblue"], ec=C["blue"], lw=1.1))
ax.text(5.0, 7.42, "vocabulary-free projection  $P_r$   (shared backbone)",
        ha="center", fontsize=7.2, color=C["blue"], weight="bold")
ax.text(5.0, 6.82, "relation-graph GNN  +  conditional propagation  (multi-source)",
        ha="center", fontsize=6.5, color=C["blue"])
arr(ax, 5.0, 8.75, 5.0, 7.98)
box(ax, 0.55, 4.75, 4.0, 0.95, "branch scores $s_i$", fc="white", ec="#555", fs=7.2)
box(ax, 5.45, 4.75, 4.0, 0.95, "witness evidence $(\\bar{w}_i,\\rho_i)$",
    fc=C["lamber"], ec=C["amber"], fs=6.7)
arr(ax, 3.2, 6.35, 2.55, 5.78); arr(ax, 6.8, 6.35, 7.45, 5.78)
ax.add_patch(FancyBboxPatch((1.1, 2.25), 7.8, 1.6,
             boxstyle="round,pad=0.012,rounding_size=0.03",
             fc=C["lred"], ec=C["red"], lw=1.5))
ax.text(5.0, 3.38, "gated logical operators   (ours, learnable)",
        ha="center", fontsize=7.4, color=C["red"], weight="bold")
ax.text(5.0, 2.75, "$\\theta_i = 0.5+3.5\\,\\sigma(\\,g(s_i^{\\max},\\bar w_i,\\rho_i)\\,)$",
        ha="center", fontsize=7.0, color=C["red"])
arr(ax, 2.55, 4.75, 3.5, 3.9, c=C["red"], lw=1.0)
arr(ax, 7.45, 4.75, 6.5, 3.9, c=C["red"], lw=1.0)
box(ax, 2.3, 0.55, 5.4, 0.95, "entity rankings   (zero-shot, any graph)",
    fc="white", ec=C["green"], fs=7.2, tc=C["green"], bold=True)
arr(ax, 5.0, 2.25, 5.0, 1.55, c=C["green"], lw=1.0)
ax.text(9.62, 3.05, "fixed ops\n$=\\ \\theta\\!\\equiv\\!1$\nspecial case",
        fontsize=5.9, color="#777", ha="center", style="italic",
        rotation=0, linespacing=1.4)
fig.tight_layout(); fig.savefig(P + "fig_arch.pdf", bbox_inches="tight"); plt.close(fig)

# ================= Fig 3: head-to-head =================
types = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"]
uq = [0.357, 0.088, 0.064, 0.355, 0.489, 0.129, 0.176, 0.127, 0.085]
sq = [0.169, 0.031, 0.043, 0.474, 0.347, 0.410, 0.507, 0.963, 0.382]
x = np.arange(len(types)); w = 0.38
fig, ax = plt.subplots(figsize=(3.5, 2.25))
ax.grid(axis="y", ls="--", lw=0.45, alpha=0.4)
ax.bar(x - w/2, uq, w, label="UltraQuery", color=C["gray"], ec="white", lw=0.5)
ax.bar(x + w/2, sq, w, label="SQA (ours)", color=C["red"], ec="white", lw=0.5)
ax.axhline(0.208, color=C["gray"], ls=":", lw=0.8)
ax.axhline(0.369, color=C["red"], ls=":", lw=0.8)
ax.text(8.55, 0.225, "avg", fontsize=5.8, color=C["gray"], ha="right")
ax.text(8.55, 0.385, "avg", fontsize=5.8, color=C["red"], ha="right")
ax.set_xticks(x); ax.set_xticklabels(types, fontsize=7.5)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylabel("MRR", fontsize=8)
ax.legend(fontsize=6.8, frameon=False, loc="upper left", handlelength=1.2)
ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(P + "fig_headtohead.pdf"); plt.close(fig)

# ================= Fig 4: boundary =================
graphs = ["FB15k-237\n(237 relations)", "WN18RR\n(11 relations)", "CoDEx-M\n(51 relations)"]
fixed = [0.121, 0.124, 0.194]; gated = [0.125, 0.091, 0.141]
gated2 = [0.147, 0.159, 0.187]; learned = [0.202, 0.091, 0.141]
x = np.arange(3); w = 0.19
fig, ax = plt.subplots(figsize=(3.5, 2.25))
ax.grid(axis="y", ls="--", lw=0.45, alpha=0.4)
b1 = ax.bar(x - 1.5*w, fixed, w, label="fixed (order-0)", color=C["gray"], ec="white", lw=0.5)
b2 = ax.bar(x - 0.5*w, gated, w, label="gated (order-1)", color=C["red"], ec="white", lw=0.5)
b3 = ax.bar(x + 0.5*w, gated2, w, label="gated (order-2)", color=C["purple"], ec="white", lw=0.5)
b4 = ax.bar(x + 1.5*w, learned, w, label="free-form", color=C["amber"], ec="white", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(graphs, fontsize=7)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylabel("zero-shot MRR", fontsize=8)
ax.legend(fontsize=6.0, frameon=False, ncol=2, loc="upper left",
          columnspacing=0.9, handlelength=1.1)
ax.set_ylim(0, 0.245)
ax.set_axisbelow(True)
ax.annotate("order-2 repairs\nthe boundary", xy=(1.10, 0.163), fontsize=6.2,
            color=C["purple"], xytext=(1.62, 0.196),
            arrowprops=dict(arrowstyle="->", lw=0.7, color=C["purple"]))
fig.tight_layout(); fig.savefig(P + "fig_boundary.pdf"); plt.close(fig)

# ================= Fig 5: theta =================
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
fig, ax = plt.subplots(figsize=(3.5, 1.95))
ax.grid(axis="y", ls="--", lw=0.45, alpha=0.4)
ax.hist(thetas, bins=25, color=C["red"], ec="white", lw=0.5, alpha=0.95)
ax.axvline(1.0, color="#333", lw=0.9, ls="--")
ymax = ax.get_ylim()[1]
ax.text(1.005, ymax*0.90, "$\\theta=1$  (product)", fontsize=6.6, color="#333")
mu = float(np.mean(thetas))
ax.axvline(mu, color=C["blue"], lw=0.9)
ax.text(mu - 0.006, ymax*0.66, f"learned mean = {mu:.2f}", fontsize=6.6,
        color=C["blue"], ha="right")
ax.set_xlabel("learned gate exponent  $\\theta$", fontsize=8)
ax.set_ylabel("branches", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(P + "fig_theta.pdf"); plt.close(fig)
print("v3 figures done")
