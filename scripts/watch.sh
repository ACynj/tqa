#!/bin/bash
# Live progress monitor. Usage: bash /mnt/ssd/tqa/scripts/watch.sh
echo "=== TQA 实验实时监控 (Ctrl+C 退出) ==="
while true; do
  clear
  echo "======== $(date '+%F %H:%M:%S') ========"
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  for f in /mnt/ssd/tqa/logs/sqa*.log /mnt/ssd/tqa/logs/pregen.log; do
    [ -f "$f" ] || continue
    echo "--- $(basename $f) (最后3行) ---"
    tail -3 "$f"
  done
  echo ""
  echo "原始日志: tail -f /mnt/ssd/tqa/logs/sqa1_fixed_seed42.log"
  sleep 10
done
