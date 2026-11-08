#!/bin/bash
# Phase 2: 15 runs = 3 models (fixed/learned/gated) x 5 seeds (42-46) x 50k steps
# Pregen missing pools first, then 4 GPU queues run jobs sequentially.
set -e
cd /mnt/ssd/tqa/src
PY=/root/miniconda3/envs/LMR/bin/python
LOG=/mnt/ssd/tqa/logs
mkdir -p $LOG /mnt/ssd/tqa/runs

echo "[$(date +%H:%M:%S)] phase2 step 1/2: pregen pools for seeds 44/45/46 ..."
$PY - <<'EOF'
import sys
sys.path.insert(0, '.')
from tqa.kgdata import load_graphs
from tqa.querygen import EPFO_TYPES
from tqa.train import get_train_pool
for seed in (44, 45, 46):
    graphs = load_graphs()
    for n, g in graphs.items():
        for qt in EPFO_TYPES:
            p = get_train_pool(g, qt, 4000, seed + 137 * (EPFO_TYPES.index(qt) + 1))
            print(f'pool {n}/{qt}/seed{seed}: {len(p)}', flush=True)
EOF

echo "[$(date +%H:%M:%S)] phase2 step 2/2: 15 training runs on 4 GPUs ..."
run_queue() {
  local gpu=$1; shift
  for spec in "$@"; do
    set -- $spec
    local ops=$1 seed=$2
    local name="p2_${ops}_s${seed}_50k"
    echo "[$(date +%H:%M:%S)] GPU$gpu START $name"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m tqa.train --ops $ops --gpu 0 \
      --steps 50000 --eval-every 5000 --seed $seed --run-name $name \
      >> $LOG/${name}.log 2>&1
    echo "[$(date +%H:%M:%S)] GPU$gpu DONE  $name"
  done
}

JOBS=(
  "fixed 42" "learned 42" "gated 42" "fixed 43"
  "learned 43" "gated 43" "fixed 44" "learned 44"
  "gated 44" "fixed 45" "learned 45" "gated 45"
  "fixed 46" "learned 46" "gated 46"
)
q0=(); q1=(); q2=(); q3=()
i=0
for spec in "${JOBS[@]}"; do
  case $((i % 4)) in
    0) q0+=("$spec") ;; 1) q1+=("$spec") ;;
    2) q2+=("$spec") ;; 3) q3+=("$spec") ;;
  esac
  i=$((i + 1))
done
echo "GPU0: ${q0[*]} | GPU1: ${q1[*]} | GPU2: ${q2[*]} | GPU3: ${q3[*]}"
run_queue 0 "${q0[@]}" &
run_queue 1 "${q1[@]}" &
run_queue 2 "${q2[@]}" &
run_queue 3 "${q3[@]}" &
wait
echo "[$(date +%H:%M:%S)] PHASE2 ALL DONE (15 runs)"
