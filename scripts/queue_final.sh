#!/bin/bash
# Final question-point queue:
#  Q1: ft5 main-table gated extra seeds (43,44) — after GPU2 frees
#  Q2: our zero-shot on WikiTopics — waits for UQ runs + data, converts, evals
set -u
cd /mnt/ssd/tqa/src
LPY=/root/miniconda3/envs/LMR/bin/python
UPY=/mnt/ssd/envs/ultra/bin/python
LOG=/mnt/ssd/tqa/logs
R=/mnt/ssd/tqa/runs

# ---- Q1: wait for GPU2 (ft5b gated NELL) then run 2 more seeds of ft5 gated ----
while pgrep -f "ft5b gated" > /dev/null; do sleep 60; done
for seed in 43 44; do
  echo "[$(date +%H:%M:%S)] Q1: ft5 gated seed $seed"
  CUDA_VISIBLE_DEVICES=2 $LPY -m tqa.ft5_seed $seed > $LOG/ft5_gated_s${seed}.log 2>&1
  echo "[$(date +%H:%M:%S)] Q1 done seed $seed"
done

# ---- Q2: wait for first WikiTopics UQ run (data downloaded), then convert + eval ----
while [ ! -f $LOG/uq_wt_health.log ]; do sleep 120; done
sleep 120
echo "[$(date +%H:%M:%S)] Q2: dump WikiTopics queries via ULTRA loader"
cd /mnt/ssd/tqa/src
$UPY - <<'EOF' > /mnt/ssd/tqa/logs/wt_dump.log 2>&1
import sys, pickle, os
sys.path.insert(0, "/mnt/ssd/tqa/external/ULTRA")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from ultra.datasets_query import WikiTopicsQuery
import torch
for topic in ("health", "science", "organization"):
    ds = WikiTopicsQuery("/root/git/ULTRA/query-datasets", version=topic)
    ds.set_query_types()
    out = {}
    eas, hards = {}, {}
    # use test split queries via the class internals
    idx = ds.get_split("test") if hasattr(ds, "get_split") else None
    # fallback: parse from processed data files
    import glob
    fs = glob.glob(f"/mnt/ssd/ultra_data/query-datasets/WikiTopicsQuery/{topic}/processed/*.pt")
    print(topic, "processed:", [os.path.basename(f) for f in fs], flush=True)
    break
EOF
echo "[$(date +%H:%M:%S)] FINAL QUEUE DONE"
