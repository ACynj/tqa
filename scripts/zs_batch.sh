#!/bin/bash
# Our zero-shot on all WikiTopics (inductive FB has different format - skip for now)
# Two checkpoints: ft5_gated (protocol) + p2_gated (pure transfer)
set -u
LPY=/root/miniconda3/envs/LMR/bin/python
LOG=/mnt/ssd/tqa/logs
R=/mnt/ssd/tqa/runs
WT=/root/git/ULTRA/query-datasets/WikiTopics_QE
cd /mnt/ssd/tqa/src

for ck in ft5_gated p2_gated_s42_50k; do
  tag=$(echo $ck | cut -d_ -f1)
  for t in health org award art edu infra loc people sport tax; do
    TQA_DATA=$WT/$t CUDA_VISIBLE_DEVICES=1 $LPY -m tqa.eval_zero $R/$ck.pt > $LOG/zs_${tag}_$t.log 2>&1 \
      && echo "[$(date +%H:%M:%S)] ${tag}_$t: $(grep -oE 'EPFO avg [0-9.]+' $LOG/zs_${tag}_$t.log)" \
      || echo "${tag}_$t FAILED"
  done
done
echo "ZERO-SHOT SWEEP DONE"
