"""3i weakness analysis + gate theta visualization (eval-only, GPU2).
- Buckets BetaE 3i (and 2i) test queries by hard-answer set size; per-bucket
  MRR for ft5_gated vs ft5_fixed vs UQ-official-type-average.
- Extracts gate theta distribution on sample 2i queries for interpretable
  cases (relation names via id2rel)."""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/ssd/tqa/src")
D = "/mnt/ssd/ultra_data/query-datasets/FB15k-237-betae"
R = "/mnt/ssd/tqa/runs"
dev = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

import tqa.eval_betae as EB
from tqa.model import TQA

q = pickle.load(open(f"{D}/test-queries.pkl", "rb"))
easy = pickle.load(open(f"{D}/test-easy-answers.pkl", "rb"))
hard = pickle.load(open(f"{D}/test-hard-answers.pkl", "rb"))
id2rel = pickle.load(open(f"{D}/id2rel.pkl", "rb"))

models = {}
for tag, ops, ck in [("ft5_gated", "gated", "ft5_gated"), ("ft5_fixed", "fixed", "ft5_fixed")]:
    m = TQA(d=64, ops=ops).to(dev)
    m.load_state_dict(torch.load(f"{R}/{ck}.pt", map_location=dev))
    m.eval()
    models[tag] = m


@torch.no_grad()
def bucket_eval(qt_struct, name):
    qs = list(q[qt_struct])[:300]
    buckets = {"small(<=2)": [], "mid(3-9)": [], "large(>=10)": []}
    for i in range(0, len(qs), 16):
        chunk = qs[i:i + 16]
        progs, _ = zip(*[EB.to_program(c) for c in chunk])
        scores = {t: m(EB.g, list(progs), dev).cpu() for t, m in models.items()}
        for j, qq in enumerate(chunk):
            ha, ea = hard[qq], easy[qq]
            filt = set(ha) | set(ea)
            mask = torch.ones(EB.g.num_ent, dtype=torch.bool)
            for x in filt:
                mask[x] = False
            n = len(ha)
            b = "small(<=2)" if n <= 2 else ("mid(3-9)" if n <= 9 else "large(>=10)")
            for t in models:
                sm = scores[t][j].clone()
                sm[~mask] = -1e9
                rrs = [1.0 / ((sm > scores[t][j, e]).sum().item() + 1) for e in ha]
                buckets[b].append((t, float(np.mean(rrs))))
    print(f"\n== {name} 按难答案集大小分桶 MRR ==")
    for b, recs in buckets.items():
        if not recs:
            continue
        line = f"  {b:<13} n={len(recs)//2:<4}"
        for t in models:
            vals = [v for tt, v in recs if tt == t]
            line += f" {t}={np.mean(vals):.3f}"
        print(line, flush=True)


bucket_eval((('e', ('r',)), ('e', ('r',)), ('e', ('r',))), "3i")
bucket_eval((('e', ('r',)), ('e', ('r',))), "2i")

# ---- theta visualization ----
print("\n== 门控θ可视化（2i样例，关系名）==")
m = models["ft5_gated"]
qs = list(q[(('e', ('r',)), ('e', ('r',)))])[:8]
with torch.no_grad():
    for qq in qs[:5]:
        prog, tgt = EB.to_program(qq)
        pops = [op for op in prog if op[0] == "P"]
        r1, r2 = pops[0][3], pops[1][3]
        s = m(EB.g, [prog], dev)
        # capture thetas via hook on ops
        thetas = []
        orig = m.ops.intersect
        def spy(sl, fl, rf=None):
            eps = 1e-4
            for s_, f_ in zip(sl, fl):
                w = (s_ / (s_.sum(dim=1, keepdim=True) + 1e-6)).unsqueeze(-1)
                fsum = (f_ * w).sum(dim=1)
                t = 0.5 + 3.5 * torch.sigmoid(
                    m.ops.gate(torch.cat([s_.max(dim=1, keepdim=True).values, fsum], dim=-1))
                    + m.ops.gate_logit)
                thetas.append(t.item())
            return orig(sl, fl, rf)
        m.ops.intersect = spy
        m(EB.g, [prog], dev)
        m.ops.intersect = orig
        n1 = id2rel.get(r1, r1) if isinstance(id2rel, dict) else r1
        n2 = id2rel.get(r2, r2) if isinstance(id2rel, dict) else r2
        print(f"  ({str(n1)[:28]} θ={thetas[0]:.2f}) ∩ ({str(n2)[:28]} θ={thetas[1]:.2f}) "
              f"{'[尖锐合取]' if max(thetas) > 1.5 else '[乘积近似]' if abs(thetas[0]-1)<0.3 else '[温和]'}",
              flush=True)
print("ANALYSIS DONE")
