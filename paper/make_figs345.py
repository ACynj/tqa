"""Figs 3-5 via figures4papers api (figlib): grouped bars + histogram,
following common-patterns.md (dynamic y-limits, white bar edges, frameless
legend) and design-theory.md palette semantics."""
import sys
sys.path.insert(0, "/mnt/ssd/tqa/paper")
from figlib import (PALETTE, FigureStyle, apply_publication_style,
                    finalize_figure, make_grouped_bar)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/mnt/ssd/tqa/paper/figs_final"

# ================= Fig 3: head-to-head (two-column grouped bars) ==========
apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.0))
types = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"]
uq = [0.357, 0.088, 0.064, 0.355, 0.489, 0.129, 0.176, 0.127, 0.085]
sq = [0.169, 0.031, 0.043, 0.474, 0.347, 0.410, 0.507, 0.963, 0.382]
fig, ax = plt.subplots(figsize=(14, 4.6))
make_grouped_bar(ax, types, [uq, sq], ["UltraQuery", "SQA (ours)"],
                 ylabel="MRR",
                 colors=[PALETTE["red_2"], PALETTE["blue_main"]],
                 annotate=True, bar_width=0.36)
ax.axhline(0.208, color=PALETTE["red_strong"], ls=":", lw=1.6)
ax.axhline(0.369, color=PALETTE["blue_main"], ls=":", lw=1.6)
ax.text(8.55, 0.215, "UQ avg", fontsize=11, color=PALETTE["red_strong"], ha="right")
ax.text(8.55, 0.376, "SQA avg", fontsize=11, color=PALETTE["blue_main"], ha="right")
ax.set_ylim(0, 1.06)
ax.legend(fontsize=13, loc="upper left", handlelength=1.2)
ax.tick_params(labelsize=13)
finalize_figure(fig, f"{OUT}/fig_headtohead", formats=["pdf", "png"], dpi=300)

# ================= Fig 4: transfer boundary (single column) ===============
apply_publication_style(FigureStyle(font_size=13, axes_linewidth=1.6))
graphs = ["FB15k-237\n(237 rel)", "WN18RR\n(11 rel)", "CoDEx-M\n(51 rel)"]
series = [[0.121, 0.124, 0.194], [0.125, 0.091, 0.141],
          [0.147, 0.159, 0.187], [0.202, 0.091, 0.141]]
labels = ["fixed (order-0)", "gated (order-1)", "gated (order-2)", "free-form"]
colors = [PALETTE["neutral"], PALETTE["red_2"], PALETTE["blue_main"], PALETTE["violet"]]
fig, ax = plt.subplots(figsize=(7, 5))
make_grouped_bar(ax, graphs, series, labels, ylabel="zero-shot MRR",
                 colors=colors, bar_width=0.19)
ax.set_ylim(0, 0.27)
ax.legend(fontsize=10.5, ncol=2, loc="upper left", handlelength=1.1,
          columnspacing=0.9)
ax.annotate("order-2 repairs\nthe boundary", xy=(1.13, 0.163), fontsize=11,
            color=PALETTE["blue_main"], xytext=(1.6, 0.21),
            arrowprops=dict(arrowstyle="->", lw=1.4, color=PALETTE["blue_main"]))
finalize_figure(fig, f"{OUT}/fig_boundary", formats=["pdf", "png"], dpi=300)

# ================= Fig 5: theta histogram =================================
apply_publication_style(FigureStyle(font_size=13, axes_linewidth=1.6))
import pickle, torch
sys.path.insert(0, "/mnt/ssd/tqa/src")
from tqa.kgdata import GraphData
from tqa.model import TQA

D = "/mnt/ssd/ultra_data/query-datasets/FB15k-237-betae"
triples = [tuple(map(int, l.split())) for l in open(f"{D}/train.txt")]
g = GraphData.__new__(GraphData)
arr = np.array(sorted(set(triples)), dtype=np.int64)
g.num_ent, g.num_rel = arr.max() + 1, int(arr[:, 1].max()) + 1
g.split_triples = {"train": arr, "valid": arr[:0], "test": arr[:0]}
g.adj_train = g._build_adj(arr); g.adj_full = g.adj_train
g.edge_index, g.edge_rel = g._edges_torch(arr)
g.rel_graph = g._build_relation_graph(arr); g._rel2edges_cache = None
g.name = "fb"

STRUCT2I = (("e", ("r",)), ("e", ("r",)))
q = pickle.load(open(f"{D}/test-queries.pkl", "rb"))
qs = list(q[STRUCT2I])[:120]

def to_2i(qq):
    (e1, c1), (e2, c2) = qq
    return [("A", 0, e1), ("P", 10, 0, c1[0]), ("A", 2, e2), ("P", 11, 2, c2[0]),
            ("I", 1, [10, 11])]

m = TQA(d=64, ops="gated")
m.load_state_dict(torch.load("/mnt/ssd/tqa/runs/ft5_gated.pt", map_location="cpu"))
m.eval()
thetas = []
orig = m.ops.intersect
def spy(sl, fl, rf=None):
    for s_, f_ in zip(sl, fl):
        w_ = (s_ / (s_.sum(dim=1, keepdim=True) + 1e-6)).unsqueeze(-1)
        fsum = (f_ * w_).sum(dim=1)
        t = 0.5 + 3.5 * torch.sigmoid(m.ops.gate(
            torch.cat([s_.max(dim=1, keepdim=True).values, fsum], -1)) + m.ops.gate_logit)
        thetas.extend(t.flatten().tolist())
    return orig(sl, fl, rf)
m.ops.intersect = spy
with torch.no_grad():
    for i in range(0, len(qs), 16):
        m(g, [to_2i(c) for c in qs[i:i + 16]], torch.device("cpu"))
m.ops.intersect = orig

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(thetas, bins=25, color=PALETTE["blue_main"], ec="white", lw=0.8)
ax.axvline(1.0, color=PALETTE["red_strong"], lw=2.0, ls="--")
ymax = ax.get_ylim()[1]
ax.text(1.004, ymax * 0.9, r"$\theta=1$ (product)", fontsize=12, color=PALETTE["red_strong"])
mu = float(np.mean(thetas))
ax.axvline(mu, color=PALETTE["teal"], lw=2.0)
ax.text(mu - 0.006, ymax * 0.68, f"learned mean = {mu:.2f}", fontsize=12,
        color=PALETTE["teal"], ha="right")
ax.set_xlabel(r"learned gate exponent $\theta$", fontsize=13)
ax.set_ylabel("branches", fontsize=13)
finalize_figure(fig, f"{OUT}/fig_theta", formats=["pdf", "png"], dpi=300)
print("figs 3-5 done via figlib; theta n =", len(thetas), "mean", round(mu, 3))
