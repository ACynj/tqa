#!/bin/bash
# Breadth push #1: FB15k as the third full dataset (UQ official + ours).
set -u
cd /mnt/ssd/tqa/src
UPY=/mnt/ssd/envs/ultra/bin/python
LPY=/root/miniconda3/envs/LMR/bin/python
LOG=/mnt/ssd/tqa/logs
UL=/mnt/ssd/tqa/external/ULTRA
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext

echo "[$(date +%H:%M:%S)] UQ official on FB15k"
cd $UL
$UPY script/run_query.py -c config/ultraquery/transductive.yaml --dataset FB15kLogicalQuery \
  --epochs 0 --bpe null --gpus [0] --bs 32 --threshold 0.0 --ultra_ckpt null \
  --qe_ckpt $UL/ckpts/ultraquery.pth > $LOG/uq_fb15k.log 2>&1
echo "[$(date +%H:%M:%S)] UQ FB15k exit=$?"

echo "[$(date +%H:%M:%S)] our zero-shot on FB15k (all three executors)"
cd /mnt/ssd/tqa/src
TQA_DATA=/mnt/ssd/ultra_data/query-datasets/FB15k-betae \
  CUDA_VISIBLE_DEVICES=0 $LPY -m tqa.eval_betae > $LOG/zeroshot_fb15k.log 2>&1

echo "[$(date +%H:%M:%S)] our ft5 (fixed, gated) on FB15k"
export TQA_DATA=/mnt/ssd/ultra_data/query-datasets/FB15k-betae
TQA_TAG=_fb15k CUDA_VISIBLE_DEVICES=0 $LPY -m tqa.ft5 fixed > $LOG/ft5_fixed_fb15k.log 2>&1
TQA_TAG=_fb15k CUDA_VISIBLE_DEVICES=0 $LPY -m tqa.ft5 gated > $LOG/ft5_gated_fb15k.log 2>&1
echo "[$(date +%H:%M:%S)] BREADTH-FB15K DONE"
