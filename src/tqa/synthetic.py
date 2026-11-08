"""Synthetic two-world T2 experiment (queued after ft3 on GPU1).
Worlds W1/W2 share identical anchor-local fuzzy structure for the ambiguous
relation r_x, but r_x has different GLOBAL co-occurrence signatures.
Prediction: score-factored (fixed) executors collide -> ~50% ceiling;
witness-aware (gated) separate -> >50%. Script builds worlds, trains both
models briefly, reports accuracy."""
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/ssd/tqa/src")
from tqa.kgdata import GraphData
from tqa.model import TQA

dev = torch.device("cuda:1")


def build_world(seed, n_blocks=40):
    """Each block: anchor a, targets t1..t4, decoys. r_a links a->{t1,t2,t3,t4}.
    Ambiguous r_x from a hits {t1,t2} in world-1, {t3,t4} in world-2 blocks.
    Which world a block is in is signalled ONLY by r_x's co-occurrence with
    marker relations m1 (world1) / m2 (world2) elsewhere in the graph."""
    rng = random.Random(seed)
    ents, rels = set(), set()
    tri = []
    N = 0

    def new_e():
        nonlocal N
        N += 1
        ents.add(N - 1)
        return N - 1

    for b in range(n_blocks):
        a = new_e()
        ts = [new_e() for _ in range(4)]
        for t in ts:
            tri.append((a, 0, t))           # r_a: a -> {t1..t4}
        world = rng.random() < 0.5
        hit = ts[:2] if world else ts[2:]   # r_x true extension
        for t in hit:
            tri.append((a, 1, t))           # r_x
        m = 2 if world else 3               # marker co-occurrence
        z1, z2 = new_e(), new_e()
        tri.append((z1, m, z2))             # r_m co-occurs with r_x via type
    arr = np.array(sorted(set(tri)), dtype=np.int64)
    g = GraphData.__new__(GraphData)
    g.name = f"synth{seed}"
    g.num_ent = N
    g.num_rel = 4
    g.split_triples = {"train": arr, "valid": arr[:0], "test": arr[:0]}
    g.adj_train = g._build_adj(arr)
    g.adj_full = g.adj_train
    g.edge_index, g.edge_rel = g._edges_torch(arr)
    g.rel_graph = g._build_relation_graph(arr)
    g._rel2edges_cache = None
    return g


train_g = build_world(1)
test_g = build_world(2)   # disjoint entities, same construction rule


def make_queries(g, n=200):
    qs, ans = [], []
    rng = random.Random(7)
    adj_a, adj_x = g.adj_train[0], g.adj_train[1]
    for _ in range(n):
        a = rng.choice(list(adj_a.keys()))
        ta, tx = set(adj_a[a]), set(adj_x.get(a, set()))
        inter = ta & tx
        if not inter or len(inter) == len(ta):
            continue
        qs.append([("A", 0, a), ("P", 10, 0, 0), ("P", 11, 0, 1), ("I", 1, [10, 11])])
        ans.append(inter)
    return qs, ans


def acc(model, g, qs, ans):
    model.eval()
    hit = 0
    with torch.no_grad():
        for i in range(0, len(qs), 16):
            s = model(g, qs[i:i + 16], dev).cpu()
            for j in range(s.shape[0]):
                if max(ans[i + j], key=lambda e: s[j, e]) in ans[i + j] or \
                   s[j].argmax().item() in ans[i + j]:
                    hit += 1
    return hit / max(len(qs), 1)


for ops in ("fixed", "gated"):
    m = TQA(d=32, ops=ops, proj_layers=2, rel_layers=3).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    qs, ans = make_queries(train_g, 400)
    rng = random.Random(0)
    m.train()
    for step in range(3000):
        idx = rng.sample(range(len(qs)), 12)
        batch = [qs[i] for i in idx]
        s = m(train_g, batch, dev)
        loss = 0
        for j, i in enumerate(idx):
            pos = torch.tensor(sorted(ans[i]), device=dev)
            loss = loss - torch.log(s[j, pos].mean() + 1e-9)
        (loss / 12).backward()
        opt.step(); opt.zero_grad()
    tq, ta_ = make_queries(test_g, 300)
    print(f"{ops}: train acc {acc(m, train_g, qs, ans):.3f} | "
          f"UNSEEN-WORLD acc {acc(m, test_g, tq, ta_):.3f}", flush=True)
print("SYNTHETIC DONE")
