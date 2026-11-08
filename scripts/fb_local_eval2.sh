#!/bin/bash
# Clean rerun: strict GPU isolation (UQ=GPU0, gated=GPU1, fixed=GPU2), sequential datasets
QD=/root/git/ULTRA/query-datasets
UL=/mnt/ssd/tqa/external/ULTRA
PY=/root/miniconda3/envs/LMR/bin/python
UPY=/mnt/ssd/envs/ultra/bin/python
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext

for v in 113 134 150 175 217 300 550; do
  echo "[$(date +%H:%M:%S)] === ind_$v ==="
  # UQ on GPU0
  (cd $UL && CUDA_VISIBLE_DEVICES=0 $UPY script/run_query.py \
    -c config/ultraquery/inductive.yaml --dataset InductiveFB15k237Query \
    --version $v --epochs 0 --bpe null --gpus [0] --bs 32 --threshold 0.0 \
    --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth \
    > /mnt/ssd/tqa/logs/uq_ind_$v.log 2>&1) &
  # gated on GPU1
  (cd /mnt/ssd/tqa/src && TQA_DATA=$QD/$v CUDA_VISIBLE_DEVICES=1 timeout 400 $PY -m tqa.eval_zero /mnt/ssd/tqa/runs/ft5_gated.pt > /mnt/ssd/tqa/logs/zs_gated_$v.log 2>&1) &
  # fixed on GPU2
  (cd /mnt/ssd/tqa/src && TQA_OPS=fixed TQA_DATA=$QD/$v CUDA_VISIBLE_DEVICES=2 timeout 400 $PY -m tqa.eval_zero /mnt/ssd/tqa/runs/ft5_fixed.pt > /mnt/ssd/tqa/logs/zs_fixed_$v.log 2>&1) &
  wait
  echo "  UQ:    $(grep -oE 'mrr: [0-9.]+' /mnt/ssd/tqa/logs/uq_ind_$v.log | tail -1)"
  echo "  gated: $(grep -oE 'EPFO avg [0-9.]+' /mnt/ssd/tqa/logs/zs_gated_$v.log | tail -1)"
  echo "  fixed: $(grep -oE 'EPFO avg [0-9.]+' /mnt/ssd/tqa/logs/zs_fixed_$v.log | tail -1)"
done
echo "[$(date +%H:%M:%S)] FB LOCAL EVAL DONE"
