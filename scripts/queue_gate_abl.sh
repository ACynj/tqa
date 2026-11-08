#!/bin/bash
# Gate-input ablation queue: waits for gated2-LOO queue to finish, then runs
# 3 ablation variants (50k each) on free GPUs.
set -u
cd /mnt/ssd/tqa/src
PY=/root/miniconda3/envs/LMR/bin/python
LOG=/mnt/ssd/tqa/logs

# wait until all 6 gated2-LOO results exist
while [ $(ls /mnt/ssd/tqa/runs/loo_gated2_ex_*_results.json 2>/dev/null | wc -l) -lt 6 ]; do sleep 120; done

run_abl() { # gpu ablate seed
  local name="abl_${2}_s${3}"
  echo "[$(date +%H:%M:%S)] GPU$1 START $name"
  CUDA_VISIBLE_DEVICES=$1 GATED_ABLATE=$2 $PY -m tqa.train --ops gated --gpu 0 \
    --steps 50000 --eval-every 25000 --seed $3 --run-name $name >> $LOG/${name}.log 2>&1
  echo "[$(date +%H:%M:%S)] GPU$1 DONE $name"
}
run_abl 1 noscore 42 & run_abl 3 nowitness 42 &
wait
run_abl 1 noscore 43 & run_abl 3 nowitness 43 &
wait
echo "[$(date +%H:%M:%S)] GATE ABLATION DONE (4 runs)"
