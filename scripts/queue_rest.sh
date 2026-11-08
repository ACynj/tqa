#!/bin/bash
# Remaining gated2 LOO queue: after current runs free up GPUs.
# GPU1: codex s42 -> fb s43   GPU3: wn s43 -> codex s43
cd /mnt/ssd/tqa/src
PY=/root/miniconda3/envs/LMR/bin/python
LOG=/mnt/ssd/tqa/logs
run() { # gpu exclude seed tag
  CUDA_VISIBLE_DEVICES=$1 $PY -m tqa.train3 --ops gated2 --gpu 0 --steps 50000 \
    --eval-every 25000 --seed $3 --exclude-graph $2 --eval-graph $2 \
    --run-name loo_gated2_ex_${2}_s${3} > $LOG/loo_gated2_${4}.log 2>&1
  echo "[$(date +%H:%M:%S)] gated2-LOO $2 s$3 done"
}
# wait for the two currently running LOO jobs to finish
while [ $(ls /mnt/ssd/tqa/runs/loo_gated2_ex_*_results.json 2>/dev/null | wc -l) -lt 2 ]; do sleep 60; done
run 1 codex-m 42 cx42 &
run 3 wn18rr 43 wn43 &
wait
run 1 fb15k237 43 fb43 &
run 3 codex-m 43 cx43 &
wait
echo "[$(date +%H:%M:%S)] ALL GATED2-LOO DONE"
