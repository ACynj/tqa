"""Standalone theta histogram (CPU, no heavy imports)."""
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/ssd/tqa/src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"blue_main": "#0F4D92", "red_strong": "#B64342", "teal": "#42949E"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.0,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.0, "pdf.fonttype": 42, "figure.dpi": 300,
                     "legend.frameon": False})

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

q = pickle.load(open(f"{D}/test-queries.pkl", "rb"))
STRUCT2I = (("e", ("r",)), ("e", ("r",)))
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
        m(g, [to_2i(c) for c in qs[i:i+16]], torch.device("cpu"))
m.ops.intersect = orig

fig, ax = plt.subplots(figsize=(3.5, 1.95))
ax.hist(thetas, bins=25, color=PALETTE["blue_main"], ec="white", lw=0.5, alpha=0.95)
ax.axvline(1.0, color=PALETTE["red_strong"], lw=1.2, ls="--")
ymax = ax.get_ylim()[1]
ax.text(1.006, ymax * 0.88, r"$\theta=1$ (product)", fontsize=7, color=PALETTE["red_strong"])
mu = float(np.mean(thetas))
ax.axvline(mu, color=PALETTE["teal"], lw=1.2)
ax.text(mu - 0.007, ymax * 0.62, f"learned mean {mu:.2f}", fontsize=7,
        color=PALETTE["teal"], ha="right")
ax.set_xlabel(r"learned gate exponent $\theta$", fontsize=8.5)
ax.set_ylabel("branches", fontsize=8.5)
ax.tick_params(labelsize=7.5)
import os
os.makedirs("/mnt/ssd/tqa/paper/figs_final", exist_ok=True)
fig.savefig("/mnt/ssd/tqa/paper/figs_final/fig_theta.pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig("/mnt/ssd/tqa/paper/figs_final/fig_theta.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
print("theta done:", len(thetas), "mean", round(mu, 3))
