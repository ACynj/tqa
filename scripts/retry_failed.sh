#!/bin/bash
# After current workers finish, rerun any dataset whose log lacks a result.
set -u
LOG=/mnt/ssd/tqa/logs
UPY=/mnt/ssd/envs/ultra/bin/python
UL=/mnt/ssd/tqa/external/ULTRA
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext
while pgrep -f "run_query.py" > /dev/null; do sleep 120; done
for t in health science organization award art education infrastructure location people sport taxonomy; do
  grep -q "mrr" $LOG/uq_wt_$t.log 2>/dev/null || {
    echo "[$(date +%H:%M:%S)] retry wt:$t"
    cd $UL && $UPY script/run_query.py -c config/ultraquery/inductive.yaml \
      --dataset WikiTopicsQuery --version $t --epochs 0 --bpe null --gpus [1] --bs 32 \
      --threshold 0.0 --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth > $LOG/uq_wt_$t.log 2>&1
  }
done
for v in 106 133 160 200 266 333 400 500 550; do
  grep -q "mrr" $LOG/uq_ind_$v.log 2>/dev/null || {
    echo "[$(date +%H:%M:%S)] retry ind:$v"
    cd $UL && $UPY script/run_query.py -c config/ultraquery/inductive.yaml \
      --dataset InductiveFB15k237Query --version $v --epochs 0 --bpe null --gpus [1] --bs 32 \
      --threshold 0.0 --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth > $LOG/uq_ind_$v.log 2>&1
  }
done
echo "[$(date +%H:%M:%S)] RETRY PASS DONE"
