#!/bin/bash
# Parallel dataset sweep: 3 GPUs after gated2-LOO finishes; ablation postponed.
set -u
LOG=/mnt/ssd/tqa/logs
UPY=/mnt/ssd/envs/ultra/bin/python
UL=/mnt/ssd/tqa/external/ULTRA
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext

uq() { # gpu dataset version tag
  cd $UL && $UPY script/run_query.py -c config/ultraquery/inductive.yaml \
    --dataset $2 --version $3 --epochs 0 --bpe null --gpus [$1] --bs 32 \
    --threshold 0.0 --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth \
    > $LOG/uq_$4.log 2>&1 || echo "$4 failed"
  echo "[$(date +%H:%M:%S)] $4 done"
}

while pgrep -f "train3 --ops gated2" > /dev/null; do sleep 60; done

worker1() { for t in science organization award art education; do uq 1 WikiTopicsQuery $t wt_$t; done; }
worker2() { for t in infrastructure location people sport taxonomy; do uq 3 WikiTopicsQuery $t wt_$t; done; }
worker3() {
  for v in 106 133 160 200 266 333 400 500 550; do uq 2 InductiveFB15k237Query $v ind_$v; done
}
worker1 & worker2 & worker3 &
wait
echo "[$(date +%H:%M:%S)] PARALLEL DATASET SWEEP DONE"
