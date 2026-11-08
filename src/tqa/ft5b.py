"""ft5: fine-tune with UltraQuery's EXACT official loss (score-as-logit,
pos/neg balanced weighted BCE, lr 5e-4), initialized from the LP-only
pretrained checkpoint (replicating their ultra_3g init path).
Usage: python -m tqa.ft5 {fixed|gated|learned}"""
import os as _o2
import pickle
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/ssd/tqa/src")
from tqa.kgdata import GraphData
from tqa.model import TQA
import tqa.eval_betae as EB

OPS = sys.argv[1] if len(sys.argv) > 1 else "fixed"
import os as _os
D = _os.environ.get("TQA_DATA", "/mnt/ssd/ultra_data/query-datasets/FB15k-237-betae")
R = "/mnt/ssd/tqa/runs"
dev = torch.device("cuda:0")
g = EB.g

qtr = pickle.load(open(f"{D}/train-queries.pkl", "rb"))
atr = pickle.load(open(f"{D}/train-answers.pkl", "rb"))

TRAIN_STRUCTS = [('e', ('r',)), ('e', ('r', 'r')), ('e', ('r', 'r', 'r')),
                 (('e', ('r',)), ('e', ('r',))), (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
                 (('e', ('r',)), ('e', ('r', 'n'))), (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))),
                 ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)),
                 (('e', ('r', 'r')), ('e', ('r', 'n'))),
                 (('e', ('r', 'r', 'n')), ('e', ('r',)))]

pools = {}
for st in TRAIN_STRUCTS:
    lst = []
    for qq in list(qtr[st])[:2000]:
        ans = set(atr.get(qq, set()))
        if ans:
            lst.append((qq, ans))
    pools[st] = lst
print(f"ft5_{OPS} [{D}]: pools ready", flush=True)

m = TQA(d=64, ops=OPS).to(dev)
m.load_state_dict(torch.load(f"{R}/lp_only_s42.pt", map_location=dev), strict=False)
opt = torch.optim.AdamW(m.parameters(), lr=float(_o2.environ.get("FT_LR", "5e-4")))
rng = random.Random(0)
m.train()

for step in range(int(_o2.environ.get("FT_STEPS", "10000"))):
    st = rng.choice(list(pools.keys()))
    items = rng.sample(pools[st], min(32, len(pools[st])))
    progs, _ = zip(*[EB.to_program(qq) for qq, _ in items])
    scores = m(g, list(progs), dev)  # in [0,1], used AS LOGITS (their recipe)
    target = torch.zeros_like(scores)
    for bi, (_, ans) in enumerate(items):
        idx = torch.tensor(sorted(ans), device=dev, dtype=torch.long)
        target[bi, idx] = 1.0
    loss_unred = F.binary_cross_entropy_with_logits(scores, target, reduction="none")
    is_pos, is_neg = target > 0.5, target <= 0.5
    n_pos, n_neg = is_pos.sum(-1), is_neg.sum(-1)
    w = torch.zeros_like(scores)
    w[is_pos] = (1.0 / n_pos.float()).repeat_interleave(n_pos)
    w[is_neg] = (1.0 / n_neg.float()).repeat_interleave(n_neg)
    loss = (loss_unred * w).sum(-1) / w.sum(-1)
    loss = loss.mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
    opt.step()
    if step % 1000 == 0:
        print(f"ft5_{OPS} step {step} loss {loss.item():.4f}", flush=True)

torch.save(m.state_dict(), f"{R}/ft5_{OPS}{_os.environ.get('TQA_TAG','')}.pt")
res = EB.evaluate(m)
print(f"\n== ft5_{OPS} ==")
epfo, neg = [], []
for t in ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u-DNF", "up-DNF"]:
    print(f"  {t:<8} {res[t][0]:.4f}"); epfo.append(res[t][0])
for t in ["2in", "3in", "inp", "pin", "pni"]:
    print(f"  {t:<8} {res[t][0]:.4f}"); neg.append(res[t][0])
print(f"  EPFO {np.mean(epfo):.4f} (UQ 0.2077) | NEG {np.mean(neg):.4f} (UQ 0.0953)", flush=True)
