import os, sys, pickle
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/tqa/src")
D = os.environ.get("TQA_DATA")
dev = torch.device("cuda:0")
from tqa.kgdata import GraphData
from tqa.model import TQA
triples = [tuple(map(int, l.split())) for l in open(os.path.join(D, "train_graph.txt"))]
g = GraphData.__new__(GraphData)
arr = np.array(sorted(set(triples)), dtype=np.int64)
g.num_ent = arr.max() + 1; g.num_rel = int(arr[:, 1].max()) + 1
g.split_triples = {"train": arr, "valid": arr[:0], "test": arr[:0]}
g.adj_train = g._build_adj(arr); g.adj_full = g.adj_train
g.edge_index, g.edge_rel = g._edges_torch(arr)
g.rel_graph = g._build_relation_graph(arr); g._rel2edges_cache = None
q = pickle.load(open(os.path.join(D, "test_queries.pkl"), "rb"))
hard_raw = pickle.load(open(os.path.join(D, "test_answers_hard.pkl"), "rb"))
easy_raw = pickle.load(open(os.path.join(D, "test_answers_easy.pkl"), "rb"))
easy, hard = {}, {}
for _, qm in easy_raw.items():
    if isinstance(qm, dict): easy.update(qm)
for _, qm in hard_raw.items():
    if isinstance(qm, dict): hard.update(qm)
# reuse parser from eval_zero
def to_prog(qq):
    prog = []; ctr = [100]
    def newv(): ctr[0] += 1; return ctr[0]
    def parse(node):
        if isinstance(node[0], int) and node[0] >= 0 and isinstance(node[1], tuple):
            e, chain = node; v = newv(); prog.append(("A", v, e))
            for step in chain:
                if step == "n" or step == -2:
                    vn = newv(); prog.append(("N", vn, v)); v = vn
                else:
                    vn = newv(); prog.append(("P", vn, v, step)); v = vn
            return v
        subs = []; is_union = False
        for s in node:
            if isinstance(s, tuple) and len(s) == 1 and s[0] in (-1, "u"): is_union = True
            elif isinstance(s, tuple) and len(s) == 1 and s[0] in ("n", -2): continue
            else: subs.append(parse(s))
        out = newv(); prog.append(("U" if is_union else "I", out, subs)); return out
    def top(node):
        if isinstance(node[0], int) and node[0] >= 0: return parse(node)
        if len(node) == 2 and isinstance(node[1], tuple) and node[1] and all(isinstance(x, int) and x >= 0 for x in node[1]):
            base = parse(node[0]); v = base
            for r in node[1]:
                vn = newv(); prog.append(("P", vn, v, r)); v = vn
            return v
        return parse(node)
    return prog, top(qq)

m = TQA(d=64, ops="fixed").to(dev)
sd = torch.load(sys.argv[1], map_location=dev)
# load with strict=False to skip missing gate params
m.load_state_dict(sd, strict=False)
m.eval()
epfo_types = [
    ('e', ('r',)), ('e', ('r', 'r')), ('e', ('r', 'r', 'r')),
    (('e', ('r',)), ('e', ('r',))), (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
    ((('e', ('r',)), ('e', ('r',))), ('r',)),
    (('e', ('r', 'r')), ('e', ('r',))),
    (('e', ('r',)), ('e', ('r',)), ('u',)),
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)),
]
rrs = []
with torch.no_grad():
    for struct in epfo_types:
        if struct not in q: continue
        qs = list(q[struct])[:150]
        for i in range(0, len(qs), 16):
            chunk = qs[i:i+16]
            progs, _ = zip(*[to_prog(c) for c in chunk])
            scores = m(g, list(progs), dev).cpu()
            for j, qq in enumerate(chunk):
                ha = hard[qq]; ea = easy.get(qq, set())
                filt = set(ha) | set(ea)
                mask = torch.ones(g.num_ent, dtype=torch.bool)
                for x in filt: mask[x] = False
                sm = scores[j].clone(); sm[~mask] = -1e9
                for e in ha:
                    t = scores[j, e]
                    rrs.append(1.0 / ((sm > t).sum().item() + 1))
print(f"EPFO avg {np.mean(rrs):.4f} (n={len(rrs)})")
