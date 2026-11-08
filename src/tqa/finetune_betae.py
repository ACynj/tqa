"""Fine-tune our executors on BetaE FB15k-237 training queries (UltraQuery
protocol: 10 patterns), then evaluate all 14 types. Init from p2 ckpts."""
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/ssd/tqa/src")
from tqa.kgdata import GraphData
from tqa.model import TQA
from tqa.train import batch_loss

D = "/mnt/ssd/ultra_data/query-datasets/FB15k-237-betae"
R = "/mnt/ssd/tqa/runs"
dev = torch.device("cuda:0")

triples = [tuple(map(int, l.split())) for l in open(f"{D}/train.txt")]
g = GraphData.__new__(GraphData)
arr = np.array(sorted(set(triples)), dtype=np.int64)
g.num_ent, g.num_rel = arr.max() + 1, int(arr[:, 1].max()) + 1
g.split_triples = {"train": arr, "valid": arr[:0], "test": arr[:0]}
g.adj_train = g._build_adj(arr); g.adj_full = g.adj_train
g.edge_index, g.edge_rel = g._edges_torch(arr)
g.rel_graph = g._build_relation_graph(arr); g._rel2edges_cache = None

TRAIN_STRUCTS = [('e', ('r',)), ('e', ('r', 'r')), ('e', ('r', 'r', 'r')),
                 (('e', ('r',)), ('e', ('r',))), (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
                 (('e', ('r',)), ('e', ('r', 'n'))), (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))),
                 ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)),
                 (('e', ('r', 'r')), ('e', ('r', 'n'))),
                 (('e', ('r', 'r', 'n')), ('e', ('r',)))]
qtr = pickle.load(open(f"{D}/train-queries.pkl", "rb"))
etr = pickle.load(open(f"{D}/train-answers.pkl", "rb"))
htr = etr

sys.path.insert(0, "/mnt/ssd/tqa/src")
import tqa.eval_betae as EB  # reuses to_program/evaluate/NAME (module eval runs once, ok)


def train(ops, init_ckpt, name, steps=10000):
    m = TQA(d=64, ops=ops).to(dev)
    m.load_state_dict(torch.load(f"{R}/{init_ckpt}.pt", map_location=dev))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    pools = {}
    for st in TRAIN_STRUCTS:
        lst = []
        from tqa.querygen import _run_program
        uni = set(range(g.num_ent))
        for qq in list(qtr[st])[:2000]:
            prog, tgt = EB.to_program(qq)
            ans = _run_program(prog, g.adj_train, uni).get(tgt, set())
            if ans:
                lst.append((qq, {"answers": ans}))
        pools[st] = lst
    print(f"{name}: {len(pools)} train queries", flush=True)
    import random
    rng = random.Random(0)
    m.train()
    for step in range(steps):
        st = rng.choice(list(pools.keys()))
        items = rng.sample(pools[st], min(32, len(pools[st])))
        progs, tgts = zip(*[EB.to_program(qq) for qq, _ in items])
        scores = m(g, list(progs), dev)
        tgt_mat = torch.zeros_like(scores)
        for bi, (_, it) in enumerate(items):
            idx = torch.tensor(sorted(it["answers"]), device=dev, dtype=torch.long)
            tgt_mat[bi, idx] = 1.0
        loss = torch.nn.functional.binary_cross_entropy(
            scores.clamp(1e-6, 1 - 1e-6), tgt_mat)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 500 == 0:
            print(f"{name} step {step} loss {loss.item():.3f}", flush=True)
    torch.save(m.state_dict(), f"{R}/{name}.pt")
    res = EB.evaluate(m)
    return res


for ops, ck, nm in [("fixed", "p2_fixed_s42_50k", "ft4_fixed"),
                    ("gated", "p2_gated_s42_50k", "ft4_gated"),
                    ("learned", "p2_learned_s42_50k", "ft4_learned")]:
    res = train(ops, ck, nm)
    print(f"\n== {nm} ==")
    epfo, neg = [], []
    for t in ["1p","2p","3p","2i","3i","ip","pi","2u-DNF","up-DNF"]:
        print(f"  {t:<8} {res[t][0]:.4f}"); epfo.append(res[t][0])
    for t in ["2in","3in","inp","pin","pni"]:
        print(f"  {t:<8} {res[t][0]:.4f}"); neg.append(res[t][0])
    print(f"  EPFO {np.mean(epfo):.4f} (UQ 0.2077) | NEG {np.mean(neg):.4f} (UQ 0.0953)", flush=True)
