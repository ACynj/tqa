#!/bin/bash
# Morning retry: verify zip integrity before extracting, longer waits
QD=/root/git/ULTRA/query-datasets
UL=/mnt/ssd/tqa/external/ULTRA
PY=/root/miniconda3/envs/LMR/bin/python
UPY=/mnt/ssd/envs/ultra/bin/python
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext

verify_zip() {
  [ -f "$1" ] || return 1
  $PY -c "
import zipfile, sys
try:
    z = zipfile.ZipFile('$1')
    sys.exit(0 if z.testzip() is None and len(z.namelist()) > 2 else 1)
except: sys.exit(1)
" 2>/dev/null
}

for attempt in $(seq 1 12); do
  remaining=0
  for v in 133 160 200 266 333 400 500; do
    [ -f "$QD/$v/train_graph.txt" ] || remaining=$((remaining+1))
  done
  [ "$remaining" -eq 0 ] && echo "ALL DONE" && break
  echo "[$(date +%H:%M:%S)] attempt $attempt, $remaining remaining"
  for v in 133 160 200 266 333 400 500; do
    if [ ! -f "$QD/$v/train_graph.txt" ]; then
      echo "  downloading $v..."
      curl -sL --max-time 300 -H "User-Agent: Mozilla/5.0" \
        "https://zenodo.org/record/7306046/files/$v.zip" -o "$QD/$v.zip"
      if verify_zip "$QD/$v.zip"; then
        cd $QD && unzip -qo $v.zip
        echo "  $v: verified & extracted"
        # UQ eval
        cd $UL && $UPY script/run_query.py \
          -c config/ultraquery/inductive.yaml --dataset InductiveFB15k237Query \
          --version $v --epochs 0 --bpe null --gpus [1] --bs 32 --threshold 0.0 \
          --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth \
          > /mnt/ssd/tqa/logs/uq_ind_$v.log 2>&1
        # our evals
        cd /mnt/ssd/tqa/src
        echo -n "  gated_$v: " >> /mnt/ssd/tqa/logs/morning_evals.log
        TQA_DATA=$QD/$v CUDA_VISIBLE_DEVICES=2 timeout 300 $PY -m tqa.eval_zero /mnt/ssd/tqa/runs/ft5_gated.pt 2>/dev/null | grep "EPFO" >> /mnt/ssd/tqa/logs/morning_evals.log
        echo -n "  fixed_$v: " >> /mnt/ssd/tqa/logs/morning_evals.log
        TQA_OPS=fixed TQA_DATA=$QD/$v CUDA_VISIBLE_DEVICES=3 timeout 300 $PY -m tqa.eval_zero /mnt/ssd/tqa/runs/ft5_fixed.pt 2>/dev/null | grep "EPFO" >> /mnt/ssd/tqa/logs/morning_evals.log
        sleep 60
      else
        rm -f "$QD/$v.zip"
        echo "  $v: still blocked"
      fi
    fi
  done
  [ "$remaining" -gt 0 ] && sleep 1800
done
echo "[$(date +%H:%M:%S)] MORNING RETRY DONE"
