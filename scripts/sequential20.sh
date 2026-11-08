#!/bin/bash
# Strictly sequential: one dataset at a time (single-writer, no cache races).
set -u
LOG=/mnt/ssd/tqa/logs
UPY=/mnt/ssd/envs/ultra/bin/python
UL=/mnt/ssd/tqa/external/ULTRA
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext
uq() {
  cd $UL && $UPY script/run_query.py -c config/ultraquery/inductive.yaml \
    --dataset $2 --version $3 --epochs 0 --bpe null --gpus [1] --bs 32 \
    --threshold 0.0 --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth \
    > $LOG/uq_$4.log 2>&1 || echo "$4 FAILED"
  echo "[$(date +%H:%M:%S)] $4: $(grep -oE 'mrr: [0-9.]+' $LOG/uq_$4.log | tail -1)"
}
for t in science organization award art education infrastructure location people sport taxonomy; do
  uq 1 WikiTopicsQuery $t wt_$t
done
for v in 106 133 160 200 266 333 400 500 550; do
  uq 1 InductiveFB15k237Query $v ind_$v
done
echo "[$(date +%H:%M:%S)] SEQUENTIAL20 ALL DONE"
