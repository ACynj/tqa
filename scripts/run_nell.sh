#!/bin/bash
# Supplementary experiments (self-running, no agent needed):
#  A) NELL995 head-to-head: UQ official + our ft5 fixed/gated (sequential GPU0)
#  B) WikiTopics-CLQA: UQ official on 3 graphs (auto-download, sequential GPU0)
set -u
cd /mnt/ssd/tqa/src
UPY=/mnt/ssd/envs/ultra/bin/python
LPY=/root/miniconda3/envs/LMR/bin/python
LOG=/mnt/ssd/tqa/logs
UL=/mnt/ssd/tqa/external/ULTRA
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext

echo "[$(date +%H:%M:%S)] A1: UQ official on NELL995"
cd $UL
$UPY script/run_query.py -c config/ultraquery/transductive.yaml --dataset NELL995LogicalQuery \
  --epochs 0 --bpe null --gpus [0] --bs 32 --threshold 0.0 --ultra_ckpt null \
  --qe_ckpt $UL/ckpts/ultraquery.pth > $LOG/uq_nell.log 2>&1 || echo "UQ NELL failed"
echo "[$(date +%H:%M:%S)] A2: our ft5 on NELL"
cd /mnt/ssd/tqa/src
for ops in fixed gated; do
  TQA_DATA=/mnt/ssd/ultra_data/query-datasets/NELL-betae TQA_TAG=_nell \
    CUDA_VISIBLE_DEVICES=0 $LPY -m tqa.ft5 $ops > $LOG/ft5_${ops}_nell.log 2>&1
  echo "[$(date +%H:%M:%S)] ft5 $ops NELL done"
done

echo "[$(date +%H:%M:%S)] B: UQ official on WikiTopics-CLQA (3 graphs)"
cd $UL
for ds in WikiTopicsQuery:health WikiTopicsQuery:science WikiTopicsQuery:organization; do
  $UPY script/run_query.py -c config/ultraquery/inductive.yaml --dataset ${ds%%:*} \
    --version ${ds##*:} --epochs 0 --bpe null --gpus [0] --bs 32 --threshold 0.0 \
    --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth \
    > $LOG/uq_wt_${ds##*:}.log 2>&1 || echo "UQ $ds failed"
  echo "[$(date +%H:%M:%S)] UQ $ds done"
done
echo "[$(date +%H:%M:%S)] SUPPLEMENTARY DONE"
