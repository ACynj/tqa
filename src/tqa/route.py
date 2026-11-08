"""Inference-time operator/projection routing: fuse fixed-trained and
gated-trained executors, weighted by the target graph's relation-vocabulary
size (T5-motivated). Also reports the oracle-alpha upper bound."""
import math
import sys

import torch

sys.path.insert(0, "/mnt/ssd/tqa/src")
from tqa.kgdata import load_graphs
from tqa.model import TQA
from tqa.train import get_eval_pool, evaluate
from tqa.querygen import EPFO_TYPES

R = "/mnt/ssd/tqa/runs"
dev = torch.device("cuda:0")


class FusedModel(torch.nn.Module):
    def __init__(self, m_fixed, m_gated, alpha):
        super().__init__()
        self.mf, self.mg, self.alpha = m_fixed, m_gated, alpha

    def forward(self, graph, plans, device):
        return self.alpha * self.mf(graph, plans, device) + \
               (1 - self.alpha) * self.mg(graph, plans, device)


def alpha_rule(num_rel):
    # route to fixed (alpha=1) when the target vocabulary is small
    return min(1.0, max(0.0, (math.log2(30.0 / num_rel)) / 2.0))


for ex in ("fb15k237", "wn18rr", "codex-m"):
    graphs = load_graphs((ex,))
    g = graphs[ex]
    pools = {qt: get_eval_pool(g, qt, 500, 777) for qt in EPFO_TYPES}
    mf = TQA(d=64, ops="fixed").to(dev)
    mf.load_state_dict(torch.load(f"{R}/loo_fixed_ex_{ex}_s42.pt", map_location=dev))
    mg = TQA(d=64, ops="learned").to(dev)
    mg.load_state_dict(torch.load(f"{R}/loo_learned_ex_{ex}_s42.pt", map_location=dev))
    mf.eval(); mg.eval()
    print(f"== target {ex} (|R|={g.num_rel}, rule-alpha={alpha_rule(g.num_rel):.2f}) ==")
    for tag, model in [("fixed", mf), ("gated", mg)]:
        _, avg = evaluate(model, g, pools, dev, max_q=300)
        print(f"  {tag:<14} MRR={avg['MRR']:.4f}")
    fused = FusedModel(mf, mg, alpha_rule(g.num_rel)).to(dev)
    fused.eval()
    _, avg = evaluate(fused, g, pools, dev, max_q=300)
    print(f"  routed(rule)   MRR={avg['MRR']:.4f}")
    best = (0, 0)
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        f = FusedModel(mf, mg, a).to(dev); f.eval()
        _, avg = evaluate(f, g, pools, dev, max_q=300)
        if avg["MRR"] > best[1]:
            best = (a, avg["MRR"])
    print(f"  routed(oracle) MRR={best[1]:.4f} (alpha={best[0]})")
