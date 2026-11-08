"""Minimal zero-shot eval on any BetaE-format dataset."""
import os, sys, pickle
import numpy as np
import torch

sys.path.insert(0, "/mnt/ssd/tqa/src")
D = os.environ.get("TQA_DATA")
CKPT = sys.argv[1]
dev = torch.device("cuda:0")

from tqa.kgdata import GraphData
from tqa.model import TQA

# use test inference graph for inductive datasets; fall back to train for transductive
import glob as _g
_test = os.path.join(D, "test_inference.txt")
_train = os.path.join(D, "train_graph.txt")
if os.path.exists(_test):
    triples = [tuple(map(int, l.split())) for l in open(_test)]
    # also include train edges for richer message passing (ULTRA protocol uses inference graph only)
    triples += [tuple(map(int, l.split())) for l in open(_train)]
else:
    triples = [tuple(map(int, l.split())) for l in open(_train)]
g = GraphData.__new__(GraphData)
arr = np.array(sorted(set(triples)), dtype=np.int64)
import pickle as _pk
_mx = arr.max()
try:
    _q = _pk.load(open(os.path.join(D, "test_queries.pkl"), "rb"))
    def _scan(x):
        global _mx
        if isinstance(x, int): _mx = max(_mx, x)
        elif isinstance(x, tuple):
            for _i in x: _scan(_i)
    for _s, _ql in _q.items():
        for _qq in list(_ql)[:100]: _scan(_qq)
except: pass
# also scan hard answers for entity IDs
try:
    for _qs in hard_raw.values():
        if isinstance(_qs, dict):
            for _ans in list(_qs.values())[:50]:
                if isinstance(_ans, (list, set)):
                    for _e in _ans: _mx = max(_mx, _e)
except: pass
g.num_ent = max(_mx + 10, 14500)
g.num_rel = int(arr[:, 1].max()) + 1
g.split_triples = {"train": arr, "valid": arr[:0], "test": arr[:0]}
g.adj_train = g._build_adj(arr)
g.adj_full = g.adj_train
g.edge_index, g.edge_rel = g._edges_torch(arr)
g.rel_graph = g._build_relation_graph(arr)
g._rel2edges_cache = None
g.name = os.path.basename(D)

q = pickle.load(open(os.path.join(D, "test_queries.pkl"), "rb"))
easy_raw = pickle.load(open(os.path.join(D, "test_answers_easy.pkl"), "rb"))
hard_raw = pickle.load(open(os.path.join(D, "test_answers_hard.pkl"), "rb"))
# WikiTopics nests answers by structure template; flatten to query->answers
easy, hard = {}, {}
for struct, qmap in easy_raw.items():
    if isinstance(qmap, dict):
        for qq, ans in qmap.items():
            easy[qq] = ans
    else:
        easy.update({qq: qmap for qq in [struct]})  # fallback
for struct, qmap in hard_raw.items():
    if isinstance(qmap, dict):
        for qq, ans in qmap.items():
            hard[qq] = ans

def to_prog(qq):
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
            elif isinstance(s, tuple) and len(s) == 1 and s[0] in ("n", -2):
                continue
            else:
                subs.append(parse(s))
        out = newv()
        prog.append(("U" if is_union else "I", out, subs))
        return out
    def top(node):
        if isinstance(node[0], int) and node[0] >= 0:
            return parse(node)
        if len(node) == 2 and isinstance(node[1], tuple) and node[1] and all(isinstance(x, int) and x >= 0 for x in node[1]):
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

import os as _os2
_ops = _os2.environ.get('TQA_OPS', 'gated')
m = TQA(d=64, ops=_ops).to(dev)
m.load_state_dict(torch.load(CKPT, map_location=dev))
m.eval()

epfo_types = [
    ('e', ('r',)), ('e', ('r', 'r')), ('e', ('r', 'r', 'r')),
    (('e', ('r',)), ('e', ('r',))), (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
    ((('e', ('r',)), ('e', ('r',))), ('r',)),
    (('e', ('r', 'r')), ('e', ('r',))),
    (('e', ('r',)), ('e', ('r',)), ('u',)),
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)),
]
epfo_rrs = []
with torch.no_grad():
    for struct in epfo_types:
        if struct not in q:
            continue
        qs = list(q[struct])[:150]
        for i in range(0, len(qs), 16):
            chunk = qs[i:i+16]
            progs, _ = zip(*[to_prog(c) for c in chunk])
            scores = m(g, list(progs), dev).cpu()
            for j, qq in enumerate(chunk):
                ha = hard[qq]
                ea = easy.get(qq, set())
                filt = set(ha) | set(ea)
                mask = torch.ones(g.num_ent, dtype=torch.bool)
                for x in filt:
                    mask[x] = False
                sm = scores[j].clone()
                sm[~mask] = -1e9
                for e in ha:
                    t = scores[j, e]
                    epfo_rrs.append(1.0 / ((sm > t).sum().item() + 1))

print(f"EPFO avg {np.mean(epfo_rrs):.4f} (n={len(epfo_rrs)})")
