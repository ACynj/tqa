"""Generate publication-quality figures for the TKDE paper."""
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/mnt/ssd/tqa/src")
plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6, "font.family": "serif",
    "pdf.fonttype": 42, "figure.dpi": 300})

C_UQ, C_FIXED, C_GATED, C_GATED2, C_LEARN = "#7f8c8d", "#3498db", "#c0392b", "#8e44ad", "#f39c12"

# ---- Fig: per-type head-to-head (official protocol) ----
types = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"]
uq = [0.357, 0.088, 0.064, 0.355, 0.489, 0.129, 0.176, 0.127, 0.085]
sq = [0.169, 0.031, 0.043, 0.474, 0.347, 0.410, 0.507, 0.963, 0.382]
x = np.arange(len(types)); w = 0.38
fig, ax = plt.subplots(figsize=(3.5, 2.1))
ax.bar(x - w/2, uq, w, label="UltraQuery (fine-tuned)", color=C_UQ, edgecolor="k", linewidth=0.3)
ax.bar(x + w/2, sq, w, label="SQA (ours)", color=C_GATED, edgecolor="k", linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(types, fontsize=7)
ax.set_ylabel("MRR"); ax.legend(fontsize=6.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("/mnt/ssd/tqa/paper/fig_headtohead.pdf"); plt.close(fig)

# ---- Fig: zero-shot transfer boundary + order-2 repair ----
graphs = ["FB15k-237\n(237 rel)", "WN18RR\n(11 rel)", "CoDEx-M\n(51 rel)"]
fixed = [0.121, 0.124, 0.194]
gated = [0.125, 0.091, 0.141]
gated2 = [0.147, 0.159, None]
learned = [0.202, 0.091, 0.141]
x = np.arange(3); w = 0.19
fig, ax = plt.subplots(figsize=(3.5, 2.1))
ax.bar(x - 1.5*w, fixed, w, label="fixed (order-0)", color=C_FIXED, edgecolor="k", linewidth=0.3)
ax.bar(x - 0.5*w, gated, w, label="gated (order-1)", color=C_GATED, edgecolor="k", linewidth=0.3)
ax.bar(x + 0.5*w, [v or 0 for v in gated2], w, label="gated (order-2)", color=C_GATED2, edgecolor="k", linewidth=0.3)
ax.bar(x + 1.5*w, learned, w, label="free-form", color=C_LEARN, edgecolor="k", linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(graphs, fontsize=7)
ax.set_ylabel("zero-shot MRR"); ax.legend(fontsize=6, frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.annotate("order-2 repairs\nthe boundary", xy=(1.14, 0.16), fontsize=6, color=C_GATED2,
            xytext=(1.35, 0.185), arrowprops=dict(arrowstyle="->", lw=0.6, color=C_GATED2))
fig.tight_layout(); fig.savefig("/mnt/ssd/tqa/paper/fig_boundary.pdf"); plt.close(fig)

# ---- Fig: theta distribution (data-driven from fine-tuned model) ----
import tqa.eval_betae as EB
from tqa.model import TQA
dev = torch.device("cuda:0")
m = TQA(d=64, ops="gated").to(dev)
m.load_state_dict(torch.load("/mnt/ssd/tqa/runs/ft5_gated.pt", map_location=dev))
m.eval()
D = "/mnt/ssd/ultra_data/query-datasets/FB15k-237-betae"
q = pickle.load(open(f"{D}/test-queries.pkl", "rb"))
thetas = []
struct2i = (('e', ('r',)), ('e', ('r',)))
qs = list(q[struct2i])[:120]
orig = m.ops.intersect
def spy(sl, fl, rf=None):
    for s_, f_ in zip(sl, fl):
        w_ = (s_ / (s_.sum(dim=1, keepdim=True) + 1e-6)).unsqueeze(-1)
        fsum = (f_ * w_).sum(dim=1)
        t = 0.5 + 3.5 * torch.sigmoid(
            m.ops.gate(torch.cat([s_.max(dim=1, keepdim=True).values, fsum], dim=-1)) + m.ops.gate_logit)
        thetas.extend(t.flatten().tolist())
    return orig(sl, fl, rf)
m.ops.intersect = spy
with torch.no_grad():
    for i in range(0, len(qs), 16):
        progs, _ = zip(*[EB.to_program(c) for c in qs[i:i+16]])
        m(EB.g, list(progs), dev)
m.ops.intersect = orig
fig, ax = plt.subplots(figsize=(3.5, 1.9))
ax.hist(thetas, bins=25, color=C_GATED, edgecolor="k", linewidth=0.3, alpha=0.9)
ax.axvline(1.0, color="k", lw=0.8, ls="--")
ax.text(1.01, ax.get_ylim()[1]*0.9, r"$\theta=1$ (product)", fontsize=6.5)
ax.set_xlabel(r"learned gate exponent $\theta$"); ax.set_ylabel("count")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("/mnt/ssd/tqa/paper/fig_theta.pdf"); plt.close(fig)
print("figures saved:", len(thetas), "theta samples, mean", np.mean(thetas).round(3))
