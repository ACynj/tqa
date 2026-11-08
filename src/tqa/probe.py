"""Finetune probe: isolate what destroys performance. 1p-only training,
300 steps, 4 conditions."""
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/ssd/tqa/src")
from tqa.kgdata import GraphData
from tqa.model import TQA
from tqa.train import batch_loss
from tqa.querygen import _run_program
import tqa.eval_betae as EB
import random

D = "/mnt/ssd/ultra_data/query-datasets/FB15k-237-betae"
R = "/mnt/ssd/tqa/runs"
dev = torch.device("cuda:0")
g = EB.g
qtr = pickle.load(open(f"{D}/train-queries.pkl", "rb"))
atr = pickle.load(open(f"{D}/train-answers.pkl", "rb"))
uni = set(range(g.num_ent))

STRUCT_1P = ('e', ('r',))
pool = []
for qq in list(qtr[STRUCT_1P])[:3000]:
    prog, tgt = EB.to_program(qq)
    full = set(atr.get(qq, set()))
    easy = _run_program(prog, g.adj_train, uni).get(tgt, set())
    if full or easy:
        pool.append((qq, {"answers": easy, "full": full}))
print(f"pool {len(pool)}", flush=True)


def probe(tag, lr, ans_key, steps=300, freeze_proj=False):
    m = TQA(d=64, ops="fixed").to(dev)
    m.load_state_dict(torch.load(f"{R}/p2_fixed_s42_50k.pt", map_location=dev))
    if freeze_proj:
        for p in m.proj.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(0)
    m.train()
    for _ in range(steps):
        items = rng.sample(pool, 12)
        progs, _ = zip(*[EB.to_program(qq) for qq, _ in items])
        s = m(g, list(progs), dev)
        loss = batch_loss(s, [{"answers": it[ans_key]} for _, it in items], dev, g.num_ent)
        opt.zero_grad(); loss.backward(); opt.step()
    res = EB.evaluate(m, per_type_n=120)
    print(f"{tag:<28} 1p={res['1p'][0]:.4f}  2i={res['2i'][0]:.4f}  2u={res['2u-DNF'][0]:.4f}", flush=True)


probe("P0 无训练(lr=0对照)", 0.0, "answers", steps=1)
probe("P1 easy答案 lr1e-4", 1e-4, "answers")
probe("P2 full答案 lr1e-4", 1e-4, "full")
probe("P3 easy lr1e-5", 1e-5, "answers")
probe("P4 easy 冻结投影 lr1e-4", 1e-4, "answers", freeze_proj=True)
print("PROBE DONE")
