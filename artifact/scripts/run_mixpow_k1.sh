#!/usr/bin/env bash
# results/onset_prediction_mixtral_power_k.md (feat-168): the metered cell at the CHOSEN budget.
#
# k comes from analysis/budget_calibration.py applying the argmin rule the registration fixed
# before the sweep ran; it is passed in rather than hardcoded so the number in the run is the
# number the selector produced. --batch-size 64 matches feat-166's cells, whose anchor draws,
# opponent and reward scores this arm reuses unchanged -- they do not depend on k, and re-drawing
# them would change two things at once (caution (v)).
#
# Usage: run_mixpow_k1.sh <k>
set -u
K=${1:?k}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
TAG=$(echo "$K" | tr -d '.')
LOG=output/logs/mixpow_metered_k$TAG.log
rm -f ~/v/logs/mixpow_metered_k$TAG.done ~/v/logs/mixpow_metered_k$TAG.fail

echo "[mp:metered k=$K] START $(date +%H:%M:%S)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values "$K" --trajectories-per-prompt 1 \
  --cap-factual 805 --cap-neutral 0 --cap-creative 0 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --data-dir data/bench/alpaca --max-new-tokens 200 --batch-size 64 \
  --output-dir "output/mixpow/conc_k$TAG" >> "$LOG" 2>&1
RC=$?
echo "[mp:metered k=$K] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/mixpow_metered_k$TAG.done || touch ~/v/logs/mixpow_metered_k$TAG.fail
