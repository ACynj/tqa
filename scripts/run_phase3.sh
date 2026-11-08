#!/bin/bash
# Phase 3 (overnight v2): runs automatically AFTER phase2 completes.
#   1) pools: negation (5 types x 300, fb15k237) + eval pools wn18rr/codex-m (500!)
#   2) LOO zero-shot wave A: seed42, 3 excludes x 3 ops (9 runs)
#   3) negation eval + collision analysis on p2 & LOO ckpts
#   4) LOO wave B: seed43 (9 runs) -> 2-seed zero-shot table
#   5) final report
set -u
cd /mnt/ssd/tqa/src
PY=/root/miniconda3/envs/LMR/bin/python
LOG=/mnt/ssd/tqa/logs
R=/mnt/ssd/tqa/runs

echo "[$(date +%H:%M:%S)] P3: waiting for phase2 (15 results) ..."
while true; do
  n=$(ls $R/p2_*_results.json 2>/dev/null | wc -l)
  [ "$n" -ge 15 ] && break
  sleep 60
done
echo "[$(date +%H:%M:%S)] P3: phase2 done, generating phase3 pools ..."
$PY - <<'EOF'
import sys
sys.path.insert(0, '.')
from tqa.kgdata import load_graphs
from tqa.querygen import NEG_TYPES
from tqa.train import get_eval_pool
from tqa.eval_extra import get_neg_pool
graphs = load_graphs()
for n in ("wn18rr", "codex-m"):
    g = graphs[n]
    for qt in ("1p","2p","2i","3i","ip","pi","2u","up"):
        p = get_eval_pool(g, qt, 500, 777)
        print(f'eval pool {n}/{qt}: {len(p)}', flush=True)
g = graphs["fb15k237"]
for qt in NEG_TYPES:
    p = get_neg_pool(g, qt, 300, 778)
    print(f'neg pool {qt}: {len(p)}', flush=True)
EOF

run_loo() {
  local gpu=$1; shift
  for spec in "$@"; do
    set -- $spec
    local ex=$1 ops=$2 seed=$3
    local name="loo_${ops}_ex_${ex}_s${seed}"
    echo "[$(date +%H:%M:%S)] GPU$gpu START $name"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m tqa.train3 --ops $ops --gpu 0 \
      --steps 50000 --eval-every 10000 --seed $seed --exclude-graph $ex \
      --eval-graph $ex --run-name $name >> $LOG/${name}.log 2>&1
    echo "[$(date +%H:%M:%S)] GPU$gpu DONE  $name"
  done
}

echo "[$(date +%H:%M:%S)] P3 wave A: LOO seed42 (9 runs) ..."
q0=("fb15k237 fixed 42" "codex-m learned 42"); q1=("fb15k237 learned 42" "codex-m gated 42")
q2=("fb15k237 gated 42" "wn18rr fixed 42");   q3=("wn18rr learned 42")
q0+=("wn18rr gated 42"); q1+=("codex-m fixed 42")
run_loo 0 "${q0[@]}" & run_loo 1 "${q1[@]}" & run_loo 2 "${q2[@]}" & run_loo 3 "${q3[@]}" &
wait
echo "[$(date +%H:%M:%S)] P3: negation eval + collision analysis ..."
for spec in "p2_fixed_s42_50k fixed" "p2_gated_s42_50k gated" "p2_learned_s42_50k learned" \
            "p2_fixed_s43_50k fixed" "p2_gated_s43_50k gated" "p2_learned_s43_50k learned" \
            "loo_fixed_ex_fb15k237_s42 fixed" "loo_gated_ex_fb15k237_s42 gated" \
            "loo_fixed_ex_fb15k237_s43 fixed" "loo_gated_ex_fb15k237_s43 gated"; do
  set -- $spec
  name=$1; ops=$2
  [ -f "$R/${name}.pt" ] || { echo "skip $name (no ckpt)"; continue; }
  CUDA_VISIBLE_DEVICES=0 $PY -m tqa.eval_extra --ckpt $R/${name}.pt --ops $ops \
    --graph fb15k237 >> $LOG/extra_eval.log 2>&1 && echo "[$(date +%H:%M:%S)] extra eval done: $name"
done

echo "[$(date +%H:%M:%S)] P3 wave B: LOO seed43 (9 runs) ..."
b0=("fb15k237 fixed 43" "codex-m learned 43"); b1=("fb15k237 learned 43" "codex-m gated 43")
b2=("fb15k237 gated 43" "wn18rr fixed 43");   b3=("wn18rr learned 43")
b0+=("wn18rr gated 43"); b1+=("codex-m fixed 43")
run_loo 0 "${b0[@]}" & run_loo 1 "${b1[@]}" & run_loo 2 "${b2[@]}" & run_loo 3 "${b3[@]}" &
wait

echo "[$(date +%H:%M:%S)] P3: final report ..."
$PY /mnt/ssd/tqa/scripts/final_report.py > $R/FINAL_REPORT.md 2>> $LOG/extra_eval.log
echo "[$(date +%H:%M:%S)] ALL OVERNIGHT WORK DONE -> runs/FINAL_REPORT.md"
