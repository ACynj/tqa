"""Head-to-head: our SQA executors on the official BetaE FB15k-237 queries,
same protocol as UltraQuery's evaluation."""
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/ssd/tqa/src")
from tqa.kgdata import GraphData
from tqa.model import TQA

import os as _os
D = _os.environ.get("TQA_DATA", "/mnt/ssd/ultra_data/query-datasets/FB15k-237-betae")
R = "/mnt/ssd/tqa/runs"
dev = torch.device("cuda:0")

# ---- build GraphData in BetaE id space ----
triples = []
for line in open(f"{D}/train.txt"):
    h, r, t = line.split()
    triples.append((int(h), int(r), int(t)))
g = GraphData.__new__(GraphData)
g.name = _os.path.basename(D)
import numpy as np
arr = np.array(sorted(set(triples)), dtype=np.int64)
g.num_ent = max(arr.max() + 1, 14505)
g.num_rel = int(arr[:, 1].max()) + 1
g.split_triples = {"train": arr, "valid": arr[:0], "test": arr[:0]}
g.adj_train = g._build_adj(arr)
g.adj_full = g.adj_train
g.edge_index, g.edge_rel = g._edges_torch(arr)
g.rel_graph = g._build_relation_graph(arr)
g._rel2edges_cache = None
print(f"graph: |E|={g.num_ent} |R|={g.num_rel} triples={len(arr)}")

# ---- queries ----
q = pickle.load(open(f"{D}/test-queries.pkl", "rb"))
easy = pickle.load(open(f"{D}/test-easy-answers.pkl", "rb"))
hard = pickle.load(open(f"{D}/test-hard-answers.pkl", "rb"))
NAME = {
    ('e', ('r',)): "1p", ('e', ('r', 'r')): "2p", ('e', ('r', 'r', 'r')): "3p",
    (('e', ('r',)), ('e', ('r',))): "2i",
    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): "3i",
    ((('e', ('r',)), ('e', ('r',))), ('r',)): "ip",
    (('e', ('r', 'r')), ('e', ('r',))): "pi",
    (('e', ('r',)), ('e', ('r',)), ('u',)): "2u-DNF",
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): "up-DNF",
    (('e', ('r',)), ('e', ('r', 'n'))): "2in",
    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): "3in",
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): "inp",
    (('e', ('r', 'r')), ('e', ('r', 'n'))): "pin",
    (('e', ('r', 'r', 'n')), ('e', ('r',))): "pni",
}


def to_program(qq):
    prog = []
    ctr = [100]

    def newv():
        ctr[0] += 1
        return ctr[0]

    def parse(node):
        if isinstance(node[0], int) and node[0] >= 0 and isinstance(node[1], tuple):
            e, chain = node
            v = newv()
            prog.append(("A", v, e))
            for step in chain:
                if step == "n" or step == -2:
                    vn = newv()
                    prog.append(("N", vn, v))
                    v = vn
                else:
                    vn = newv()
                    prog.append(("P", vn, v, step))
                    v = vn
            return v
        subs = []
        is_union = False
        for s in node:
            if isinstance(s, tuple) and len(s) == 1 and s[0] in (-1, "u"):
                is_union = True
            elif isinstance(s, tuple) and len(s) == 1 and s[0] == "n":
                continue  # DM marker (not in DNF protocol)
            else:
                subs.append(parse(s))
        out = newv()
        prog.append(("U" if is_union else "I", out, subs))
        return out

    def top(node):
        # top level may be (intersection_tuple, (rel_chain)) = project after ops
        if isinstance(node[0], int) and node[0] >= 0:
            return parse(node)  # plain leaf query (1p/2p/3p)
        if len(node) == 2 and isinstance(node[1], tuple) and node[1] \
                and all(isinstance(x, int) and x >= 0 for x in node[1]):
            base = parse(node[0])
            v = base
            for r in node[1]:
                vn = newv()
                prog.append(("P", vn, v, r))
                v = vn
            return v
        return parse(node)

    tgt = top(qq)
    return prog, tgt


def evaluate(model, per_type_n=300):
    res = {}
    model.eval()
    with torch.no_grad():
        for struct, name in NAME.items():
            qs = list(q[struct])[:per_type_n]
            rrs, h1, h3, h10, cnt = [], 0, 0, 0, 0
            for i in range(0, len(qs), 16):
                chunk = qs[i:i + 16]
                progs, tgts = zip(*[to_program(c) for c in chunk])
                assert len(set(tgts)) == 1
                scores = model(g, list(progs), dev).cpu()
                for j, qq in enumerate(chunk):
                    ha = hard[qq]
                    ea = easy[qq]
                    filt = set(ha) | set(ea)
                    mask = torch.ones(g.num_ent, dtype=torch.bool)
                    for x in filt:
                        mask[x] = False
                    sm = scores[j].clone()
                    sm[~mask] = -1e9
                    for e in ha:
                        t = scores[j, e]
                        rk = (sm > t).sum().item() + 1
                        rrs.append(1.0 / rk)
                        h1 += rk <= 1; h3 += rk <= 3; h10 += rk <= 10
                        cnt += 1
            res[name] = (round(float(np.mean(rrs)), 4) if rrs else 0.0,
                         round(h1 / max(cnt, 1), 4), round(h3 / max(cnt, 1), 4),
                         round(h10 / max(cnt, 1), 4))
    return res


UQ = {"1p": 0.3572, "2p": 0.0881, "3p": 0.0636, "2i": 0.3548, "3i": 0.4892,
      "ip": 0.1288, "pi": 0.1762, "2u-DNF": 0.1271, "up-DNF": 0.0847,
      "2in": 0.0978, "3in": 0.183, "inp": 0.0764, "pin": 0.0746, "pni": 0.0448}

for ck, ops in [("p2_fixed_s42_50k", "fixed"), ("p2_gated_s42_50k", "gated"),
                ("p2_learned_s42_50k", "learned")]:
    m = TQA(d=64, ops=ops).to(dev)
    m.load_state_dict(torch.load(f"{R}/{ck}.pt", map_location=dev))
    res = evaluate(m)
    print(f"\n== {ck} ==")
    epfo, neg = [], []
    for t in ["1p","2p","3p","2i","3i","ip","pi","2u-DNF","up-DNF"]:
        print(f"  {t:<8} MRR={res[t][0]:.4f} (UQ {UQ[t]:.4f})  Δ{res[t][0]-UQ[t]:+.4f}")
        epfo.append(res[t][0])
    for t in ["2in","3in","inp","pin","pni"]:
        neg.append(res[t][0])
        print(f"  {t:<8} MRR={res[t][0]:.4f} (UQ {UQ[t]:.4f})  Δ{res[t][0]-UQ[t]:+.4f}")
    print(f"  EPFO avg {np.mean(epfo):.4f} (UQ 0.2077) | NEG avg {np.mean(neg):.4f} (UQ 0.0953)")
