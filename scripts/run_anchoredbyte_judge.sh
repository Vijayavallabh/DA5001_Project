#!/usr/bin/env bash
# feat-187: the three registered judge passes, one per k, on one card, after that k's trajectories exist
# (marker ab70_k<k>.done or .fail, with a deadline). Then the scorer.
# Usage: run_anchoredbyte_judge.sh <gpu> ["0.5 0.1 2"]
set -u
GPU=${1:?gpu}; KS=${2:-0.5 0.1 2}
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
M=${AB_MARKS:-$HOME/v/logs}
for K in $KS; do
  dl=$(( $(date +%s) + 21600 ))
  until [ -e "$M/ab70_k$K.done" ] || [ -e "$M/ab70_k$K.fail" ]; do [ "$(date +%s)" -ge "$dl" ] && exit 2; sleep 60; done
  [ -e "$M/ab70_k$K.done" ] || { echo "[abj] k=$K generation failed; not judged"; continue; }
  rm -f "$M/abj_k$K.done" "$M/abj_k$K.fail"
  .venv/bin/python analysis/order_averaged_h2h.py --deecho --seed 7717 --n 64 --out results \
    --sel-dir output/phase5/sel_comma7b_64 --rewards results/selection_rewards64_comma7b.csv \
    --metered-dir output/anchoredbyte/comma7b_70b --k "$K" --anchor-dir output/phase5/sel_comma7b_64 \
    --baseline-dir output/sweep_plain --extra-dir output/phase5/imit_llama70b --extra-token=-1 \
    --extra-name risky70b --tag "ab70_k$K" > "output/logs/abj_k$K.log" 2>&1 \
    && touch "$M/abj_k$K.done" || touch "$M/abj_k$K.fail"
done
.venv/bin/python analysis/anchoredbyte_score.py --out results > output/logs/abj_score.log 2>&1
