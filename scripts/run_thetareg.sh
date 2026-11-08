#!/bin/bash
# Theta-regularized gated runs: does pulling theta toward the product point
# (robust fuzzy prior) fix zero-shot transfer on simple graphs?
set -u
cd /mnt/ssd/tqa/src
PY=/root/miniconda3/envs/LMR/bin/python
LOG=/mnt/ssd/tqa/logs
REG=0.1

run_job() {
  local gpu=$1 ex=$2 seed=$3
  local name="looreg_gated_ex_${ex}_s${seed}"
  echo "[$(date +%H:%M:%S)] GPU$gpu START $name"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m tqa.train3 --ops gated --gpu 0 \
    --steps 50000 --eval-every 10000 --seed $seed --exclude-graph $ex \
    --eval-graph $ex --theta-reg $REG --run-name $name >> $LOG/${name}.log 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE  $name"
}

run_job 1 fb15k237 42 & run_job 2 wn18rr 42 & run_job 3 codex-m 42 &
wait
run_job 1 wn18rr 43 & run_job 2 codex-m 43 &
run_job 3 fb15k237 43 &
wait
# in-domain theta-reg run (train3 with no exclusion = trains on all 3 graphs)
CUDA_VISIBLE_DEVICES=0 $PY -m tqa.train3 --ops gated --gpu 0 --steps 50000 \
  --eval-every 10000 --seed 42 --theta-reg $REG \
  --run-name p2_gatedreg_s42_50k >> $LOG/p2_gatedreg_s42_50k.log 2>&1
echo "[$(date +%H:%M:%S)] THETAREG ALL DONE"
