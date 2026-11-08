#!/bin/bash
# Our zero-shot breadth eval: ft5_gated (protocol-matched) + p2_gated (pure transfer)
# on all WikiTopics + inductive FB datasets. Runs after UQ sweep finishes on GPU0.
set -u
LPY=/root/miniconda3/envs/LMR/bin/python
LOG=/mnt/ssd/tqa/logs
R=/mnt/ssd/tqa/runs

# wait for UQ sweep to finish (all 19 datasets have mrr)
while true; do
  n=0
  for t in health org award art edu infra loc people sport tax; do
    grep -qE "mrr: [0-9]" $LOG/uq_wt_$t.log 2>/dev/null && n=$((n+1))
  done
  for v in 106 133 160 200 266 333 400 500 550; do
    grep -qE "mrr: [0-9]" $LOG/uq_ind_$v.log 2>/dev/null && n=$((n+1))
  done
  [ "$n" -ge 19 ] && break
  sleep 120
done
echo "[$(date +%H:%M:%S)] UQ sweep done, starting our zero-shot evals"

cd /mnt/ssd/tqa/src

# WikiTopics directories (from extracted zip)
WT=/root/git/ULTRA/query-datasets/WikiTopics_QE
# Inductive FB directories
for v in 106 133 160 200 266 333 400 500 550; do
  eval_one() { # ckpt_name ckpt_path dataset_dir tag gpu
    local out=$LOG/zs_$1_$4.log
    TQA_DATA=$3 CUDA_VISIBLE_DEVICES=$5 $LPY -m tqa.eval_betae_zero $2 > $out 2>&1 \
      && echo "[$(date +%H:%M:%S)] zs_$1_$4: $(grep -oE 'EPFO avg [0-9.]+' $out | tail -1)" \
      || echo "[$(date +%H:%M:%S)] zs_$1_$4 FAILED"
  }
done

# We need a lightweight eval that skips the 3-model import eval.
# Use a custom script instead. Let me create it first.
$LPY - <<'PYEOF' > /dev/null 2>&1
# create a minimal zero-shot eval script
code = '''
import os, sys, pickle
import numpy as np
import torch

sys.path.insert(0, "/mnt/ssd/tqa/src")
D = os.environ.get("TQA_DATA")
CKPT = sys.argv[1]
dev = torch.device("cuda:0")

from tqa.kgdata import GraphData
from tqa.model import TQA

triples = [tuple(map(int, l.split())) for l in open(os.path.join(D, "train.txt"))]
g = GraphData.__new__(GraphData)
arr = np.array(sorted(set(triples)), dtype=np.int64)
g.num_ent = max(arr.max() + 1, 14505)
g.num_rel = int(arr[:, 1].max()) + 1
g.split_triples = {"train": arr, "valid": arr[:0], "test": arr[:0]}
g.adj_train = g._build_adj(arr)
g.adj_full = g.adj_train
g.edge_index, g.edge_rel = g._edges_torch(arr)
g.rel_graph = g._build_relation_graph(arr)
g._rel2edges_cache = None
g.name = os.path.basename(D)

q = pickle.load(open(os.path.join(D, "test-queries.pkl"), "rb"))
easy = pickle.load(open(os.path.join(D, "test-easy-answers.pkl"), "rb"))
hard = pickle.load(open(os.path.join(D, "test-hard-answers.pkl"), "rb"))

STRUCT2I = (("e", ("r",)), ("e", ("r",)))

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

m = TQA(d=64, ops="gated").to(dev)
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
                ea = easy[qq]
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
'''
with open("/mnt/ssd/tqa/src/tqa/eval_zero.py", "w") as f:
    f.write(code)
print("eval_zero.py created")
PYEOF

# Now run all evals
echo "=== ft5_gated (protocol-matched) ==="
for t in health org award art edu infra loc people sport tax; do
  TQA_DATA=$WT/$t CUDA_VISIBLE_DEVICES=0 $LPY -m tqa.eval_zero $R/ft5_gated.pt > $LOG/zs_ft5_$t.log 2>&1 \
    && echo "ft5_$t: $(grep -oE 'EPFO avg [0-9.]+' $LOG/zs_ft5_$t.log | tail -1)" || echo "ft5_$t FAILED"
done

echo "=== p2_gated (pure transfer) ==="
for t in health org award art edu infra loc people sport tax; do
  TQA_DATA=$WT/$t CUDA_VISIBLE_DEVICES=0 $LPY -m tqa.eval_zero $R/p2_gated_s42_50k.pt > $LOG/zs_p2_$t.log 2>&1 \
    && echo "p2_$t: $(grep -oE 'EPFO avg [0-9.]+' $LOG/zs_p2_$t.log | tail -1)" || echo "p2_$t FAILED"
done

echo "[$(date +%H:%M:%S)] OUR ZERO-SHOT SWEEP DONE"
