#!/bin/bash
# Breadth push #2: our zero-shot on WikiTopics-CLQA (waits for UQ health run
# to finish downloading/processing, dumps queries to BetaE-format pickles
# via the ULTRA loader, then evaluates our executors zero-shot).
set -u
LOG=/mnt/ssd/tqa/logs
UPY=/mnt/ssd/envs/ultra/bin/python
LPY=/root/miniconda3/envs/LMR/bin/python

# wait until the UQ health run has produced its results (data ready)
while ! grep -q "mrr" $LOG/uq_wt_health.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] WikiTopics data ready; dumping queries"

cd /mnt/ssd/tqa/src
$UPY - <<'EOF' > $LOG/wt_dump.log 2>&1
import sys, os, pickle
sys.path.insert(0, "/mnt/ssd/tqa/external/ULTRA")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from ultra.datasets_query import WikiTopicsQuery
import torch

for topic in ("health", "science", "organization", "art", "education",
              "infrastructure", "location", "people", "sport", "taxonomy",
              "award"):
    try:
        root = "/root/git/ULTRA/query-datasets"
        # inspect processed dir of the health topic to learn layout once
        ds = WikiTopicsQuery(root, version=topic)
        ds.set_query_types()
        # reach into the dataset object for test queries/answers
        data = ds.data if hasattr(ds, "data") else None
        print(topic, "OK", type(ds).__name__, flush=True)
        break
    except Exception as e:
        print(topic, "ERR", repr(e)[:200], flush=True)
        break
# dump internals for format discovery
d = WikiTopicsQuery("/root/git/ULTRA/query-datasets", version="health")
print([a for a in dir(d) if not a.startswith("_")], flush=True)
EOF
echo "[$(date +%H:%M:%S)] WT DUMP PROBE DONE (format info in wt_dump.log)"
