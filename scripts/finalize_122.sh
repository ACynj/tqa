#!/bin/bash
# Finalize FB-122 once a complete data source is available.
#
# Put EITHER:
#   (a) the full official 122.zip   (md5 272d2cc1e3f98f76d02daaf066f9d653), or
#   (b) just train_graph.txt + test_inference.txt (from the graphs-only
#       Zenodo record 7306061)
# into  /home/ynj/new-pro/kgfm_fb_inductive/FB_122/incoming/
# then run:  bash /mnt/ssd/tqa/scripts/finalize_122.sh
set -eu
INC=/home/ynj/new-pro/kgfm_fb_inductive/FB_122/incoming
QD=/root/git/ULTRA/query-datasets
LPY=/root/miniconda3/envs/LMR/bin/python
UPY=/mnt/ssd/envs/ultra/bin/python
LOG=/mnt/ssd/tqa/logs

mkdir -p "$INC"
if [ -f "$INC/122.zip" ]; then
  echo "[1/5] installing full 122.zip"
  md5sum "$INC/122.zip"
  rm -rf "$QD/122" "$QD/122.zip"
  unzip -q "$INC/122.zip" -d "$QD"
  rm -f "$QD/122"/pre_filter.pt "$QD/122"/pre_transform.pt
elif [ -f "$INC/train_graph.txt" ] && [ -f "$INC/test_inference.txt" ]; then
  echo "[1/5] installing graph files into existing 122/"
  # sanity: the recovered zip already provides everything else
  cp "$INC/train_graph.txt" "$INC/test_inference.txt" "$QD/122/"
  rm -f "$QD/122"/pre_filter.pt "$QD/122"/pre_transform.pt
else
  echo "ERROR: put 122.zip or {train_graph.txt,test_inference.txt} in $INC"; exit 1
fi
# refresh the local zip from the (now complete) folder so PyG's extractor is happy
rm -f "$QD/122.zip" && (cd "$QD/122" && zip -rq ../122.zip .)

echo "[2/5] rebuild test_answers_easy.pkl (symbolic, self-checked)"
$LPY /mnt/ssd/tqa/scripts/rebuild_122_easy.py
# placeholders for files the test-only eval does not need
$LPY - <<'PY'
import pickle, os
D = "/root/git/ULTRA/query-datasets/122"
for f in ("valid_queries.pkl", "train_answers_hard.pkl"):
    p = f"{D}/{f}"
    if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
        pickle.dump({}, open(p, "wb")); print("placeholder:", f)
PY

echo "[3/5] UQ on FB-122 (dense relation-graph patch)"
cd /mnt/ssd/tqa/external/ULTRA
export PATH=/mnt/ssd/envs/ultra/bin:$PATH CUDA_HOME=/mnt/ssd/envs/ultra TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext
CUDA_VISIBLE_DEVICES=0 $UPY /mnt/ssd/tqa/scripts/uq122_patched.py > $LOG/uq_ind_122.log 2>&1 \
  || { echo "UQ FAILED"; tail -5 $LOG/uq_ind_122.log; exit 1; }
grep -E "\[EPFO\] mrr" $LOG/uq_ind_122.log | tail -1

echo "[4/5] our zero-shot executors on FB-122"
cd /mnt/ssd/tqa/src
TQA_DATA=$QD/122 CUDA_VISIBLE_DEVICES=1 timeout 900 $LPY -m tqa.eval_zero \
  /mnt/ssd/tqa/runs/ft5_gated.pt > $LOG/zs_gated_122.log 2>&1
TQA_OPS=fixed TQA_DATA=$QD/122 CUDA_VISIBLE_DEVICES=2 timeout 900 $LPY -m tqa.eval_zero \
  /mnt/ssd/tqa/runs/ft5_fixed.pt > $LOG/zs_fixed_122.log 2>&1
grep -h "EPFO avg" $LOG/zs_gated_122.log $LOG/zs_fixed_122.log

echo "[5/5] done -- recompute Table indfb/breadth and update main.tex"
