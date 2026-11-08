#!/bin/bash
QD=/root/git/ULTRA/query-datasets
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext
UL=/mnt/ssd/tqa/external/ULTRA

for attempt in $(seq 1 24); do
  all_ok=true
  for v in 133 160 200 266 333 400 500; do
    if [ ! -f "$QD/$v/train_graph.txt" ]; then
      echo "[$(date +%H:%M:%S)] attempt $attempt: downloading $v..."
      curl -sL --max-time 180 -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64)" \
        "https://zenodo.org/record/7306046/files/$v.zip" -o "$QD/$v.zip" 2>/dev/null
      sz=$(stat -c%s "$QD/$v.zip" 2>/dev/null || echo 0)
      if [ "$sz" -gt 5000 ]; then
        cd $QD && unzip -qo $v.zip 2>/dev/null
        echo "  $v OK ($sz bytes)"
        # run UQ eval
        cd $UL && /mnt/ssd/envs/ultra/bin/python script/run_query.py \
          -c config/ultraquery/inductive.yaml --dataset InductiveFB15k237Query \
          --version $v --epochs 0 --bpe null --gpus [1] --bs 32 --threshold 0.0 \
          --ultra_ckpt null --qe_ckpt $UL/ckpts/ultraquery.pth \
          > /mnt/ssd/tqa/logs/uq_ind_$v.log 2>&1
        echo "  UQ $v: $(grep -oE 'mrr: [0-9.]+' /mnt/ssd/tqa/logs/uq_ind_$v.log | tail -1)"
        # run our zero-shot
        cd /mnt/ssd/tqa/src
        echo -n "  gated_$v: "
        TQA_DATA=$QD/$v CUDA_VISIBLE_DEVICES=2 timeout 300 /root/miniconda3/envs/LMR/bin/python -m tqa.eval_zero /mnt/ssd/tqa/runs/ft5_gated.pt 2>/dev/null | grep "EPFO"
        echo -n "  fixed_$v: "
        TQA_OPS=fixed TQA_DATA=$QD/$v CUDA_VISIBLE_DEVICES=3 timeout 300 /root/miniconda3/envs/LMR/bin/python -m tqa.eval_zero /mnt/ssd/tqa/runs/ft5_fixed.pt 2>/dev/null | grep "EPFO"
      else
        echo "  $v failed ($sz bytes)"
        rm -f "$QD/$v.zip"
        all_ok=false
      fi
      sleep 30
    fi
  done
  $all_ok && break
  echo "[$(date +%H:%M:%S)] waiting 10 min before next attempt..."
  sleep 600
done
echo "ZENODO RETRY DONE"
