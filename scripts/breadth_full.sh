#!/bin/bash
# FULL BREADTH: match & exceed UltraQuery's 23 datasets.
#  - UQ official: remaining 8 WikiTopics + 9 InductiveFB15k237Query versions
#  - Ours: zero-shot on each dataset right after its data lands (converter)
# Sequential on GPU2 after the current supp chain finishes.
set -u
LOG=/mnt/ssd/tqa/logs
UPY=/mnt/ssd/envs/ultra/bin/python
LPY=/root/miniconda3/envs/LMR/bin/python
UL=/mnt/ssd/tqa/external/ULTRA
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext

# wait for the supp chain (3 wikittopics) to finish
while pgrep -f "WikiTopicsQuery --version (health|science|organization)" > /dev/null; do sleep 120; done

WT="award art education infrastructure location people sport taxonomy"
IND="106 133 160 200 266 333 400 500 550"

for t in $WT; do
  echo "[$(date +%H:%M:%S)] UQ WikiTopics:$t"
  cd $UL && $UPY script/run_query.py -c config/ultraquery/inductive.yaml \
    --dataset WikiTopicsQuery --version $t --epochs 0 --bpe null --gpus [2] --bs 32 \
    --threshold 0.0 --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth \
    > $LOG/uq_wt_$t.log 2>&1 || echo "wt:$t failed"
done
for v in $IND; do
  echo "[$(date +%H:%M:%S)] UQ InductiveFB15k237Query:$v"
  cd $UL && $UPY script/run_query.py -c config/ultraquery/inductive.yaml \
    --dataset InductiveFB15k237Query --version $v --epochs 0 --bpe null --gpus [2] --bs 32 \
    --threshold 0.0 --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth \
    > $LOG/uq_ind_$v.log 2>&1 || echo "ind:$v failed"
done

echo "[$(date +%H:%M:%S)] dumping dataset formats for our-side converter"
cd /mnt/ssd/tqa/src
$UPY - <<'EOF' > $LOG/wt_dump_full.log 2>&1
import sys, os
sys.path.insert(0, "/mnt/ssd/tqa/external/ULTRA")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from ultra.datasets_query import WikiTopicsQuery, InductiveFB15k237Query
import torch
for cls, kw in ((WikiTopicsQuery, {"version": "taxonomy"}),
                (InductiveFB15k237Query, {"version": 550})):
    try:
        ds = cls("/root/git/ULTRA/query-datasets", **kw)
        ds.set_query_types()
        attrs = [a for a in dir(ds) if not a.startswith("_")]
        print(cls.__name__, "attrs:", attrs, flush=True)
        for a in ("queries", "easy_answers", "hard_answers", "graph",
                  "train_graph", "target_graph", "data", "datasets"):
            if hasattr(ds, a):
                v = getattr(ds, a)
                print(" ", a, type(v).__name__,
                      (len(v) if hasattr(v, "__len__") else ""), flush=True)
    except Exception as e:
        print(cls.__name__, "ERR", repr(e)[:150], flush=True)
EOF
echo "[$(date +%H:%M:%S)] FULL BREADTH UQ SIDE DONE (20 datasets)"
