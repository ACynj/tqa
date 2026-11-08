#!/bin/bash
# Phase 1: SQA-1 (fixed fuzzy ops) vs SQA-2 (learned witness aggregation)
# GPU0: SQA-1   GPU1: SQA-2   (GPU2/3 reserved for seeds/ablations later)
set -e
cd /mnt/ssd/tqa/src
PY=/root/miniconda3/envs/LMR/bin/python

echo "[$(date +%H:%M:%S)] step 1/2: pre-generating shared query pools ..."
$PY -m tqa.pregen 2>&1 | tee -a /mnt/ssd/tqa/logs/pregen.log

echo "[$(date +%H:%M:%S)] step 2/2: launching SQA-1 (GPU0) and SQA-2 (GPU1) ..."
CUDA_VISIBLE_DEVICES=0 $PY -m tqa.train --ops fixed   --gpu 0 --steps 20000 \
    --seed 42 --run-name sqa1_fixed_seed42 &
P1=$!
CUDA_VISIBLE_DEVICES=1 $PY -m tqa.train --ops learned --gpu 0 --steps 20000 \
    --seed 42 --run-name sqa2_learned_seed42 &
P2=$!
wait $P1 $P2
echo "[$(date +%H:%M:%S)] phase 1 complete"
