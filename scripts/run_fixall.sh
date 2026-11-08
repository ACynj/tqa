#!/bin/bash
# Fix-all: (a) LP-only pretrain (replicates UltraQuery init path) -> ft5 with
# their EXACT official loss; (b) gated2 seeds 43-46 for the order-2 ablation.
set -u
cd /mnt/ssd/tqa/src
PY=/root/miniconda3/envs/LMR/bin/python
LOG=/mnt/ssd/tqa/logs
R=/mnt/ssd/tqa/runs

# LP-only pretrain copy (1p only)
if [ ! -f train_lponly.py ]; then
  sed 's/from tqa.querygen import EPFO_TYPES, gen_pool/from tqa.querygen import gen_pool\nEPFO_TYPES = ["1p"]/' train.py > train_lponly.py
fi

chain_gpu2() {
  echo "[$(date +%H:%M:%S)] GPU2: LP-only pretrain"
  CUDA_VISIBLE_DEVICES=2 $PY -m tqa.train_lponly --ops fixed --gpu 0 --steps 50000 \
    --eval-every 25000 --seed 42 --run-name lp_only_s42 >> $LOG/lp_only_s42.log 2>&1
  for ops in fixed gated learned; do
    echo "[$(date +%H:%M:%S)] GPU2: ft5_$ops (official loss, LP-init)"
    CUDA_VISIBLE_DEVICES=2 $PY -m tqa.ft5 $ops >> $LOG/ft5_${ops}.log 2>&1
  done
  echo "[$(date +%H:%M:%S)] GPU2 chain DONE"
}

chain_gpu1() {
  for s in 43 45; do
    CUDA_VISIBLE_DEVICES=1 $PY -m tqa.train --ops gated2 --gpu 0 --steps 50000 \
      --eval-every 25000 --seed $s --run-name p2_gated2_s${s}_50k >> $LOG/p2_gated2_s${s}_50k.log 2>&1
    echo "[$(date +%H:%M:%S)] GPU1 gated2 s$s done"
  done
}
chain_gpu3() {
  for s in 44 46; do
    CUDA_VISIBLE_DEVICES=3 $PY -m tqa.train --ops gated2 --gpu 0 --steps 50000 \
      --eval-every 25000 --seed $s --run-name p2_gated2_s${s}_50k >> $LOG/p2_gated2_s${s}_50k.log 2>&1
    echo "[$(date +%H:%M:%S)] GPU3 gated2 s$s done"
  done
}
chain_gpu2 & chain_gpu1 & chain_gpu3 &
wait
echo "[$(date +%H:%M:%S)] FIXALL DONE"
