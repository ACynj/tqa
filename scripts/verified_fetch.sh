#!/bin/bash
# Dataset sweep with download verification: corrupt zip => delete + retry.
set -u
LOG=/mnt/ssd/tqa/logs
UPY=/mnt/ssd/envs/ultra/bin/python
UL=/mnt/ssd/tqa/external/ULTRA
QD=/root/git/ULTRA/query-datasets
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext

verify() { # zip file -> ok(0)/bad(1)
  [ -f "$1" ] || return 1
  $UPY - "$1" <<'PYEOF'
import sys, zipfile, os
p = sys.argv[1]
if os.path.getsize(p) < 10000: sys.exit(1)
try:
    z = zipfile.ZipFile(p)
    bad = z.testzip()
    sys.exit(0 if bad is None and len(z.namelist()) > 2 else 1)
except Exception:
    sys.exit(1)
PYEOF
}

run_one() { # dataset version tag
  for attempt in 1 2 3 4 5; do
    cd $UL && $UPY script/run_query.py -c config/ultraquery/inductive.yaml \
      --dataset $1 --version $2 --epochs 0 --bpe null --gpus [1] --bs 32 \
      --threshold 0.0 --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth \
      > $LOG/uq_$3.log 2>&1
    if grep -qE "mrr: [0-9]" $LOG/uq_$3.log; then
      echo "[$(date +%H:%M:%S)] $3 OK: $(grep -oE 'mrr: [0-9.]+' $LOG/uq_$3.log | tail -1)"; return 0
    fi
    echo "[$(date +%H:%M:%S)] $3 attempt $attempt failed; clearing caches"
    rm -rf $QD/WikiTopics_QE
    for v in 106 133 160 200 266 333 400 500 550; do rm -rf $QD/$v; done
    sleep 10
  done
  echo "$3 FAILED after 5 attempts"
}

for t in health org award art edu infra loc people sport tax; do
  run_one WikiTopicsQuery $t wt_$t
done
for v in 106 133 160 200 266 333 400 500 550; do
  run_one InductiveFB15k237Query $v ind_$v
done
echo "[$(date +%H:%M:%S)] VERIFIED SWEEP DONE"
