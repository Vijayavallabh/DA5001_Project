#!/usr/bin/env bash
# results/onset_prediction_mixtral_power_k.md (feat-168): the calibration sweep.
#
# Reports ACTIVITY RATES ONLY. No judge runs on this, no utility is computed from it, and no band
# in the registration is readable from it -- the budget is picked by the argmin rule the
# registration fixed BEFORE this ran, so there is no free choice left at selection time.
#
# One card per k, no PID capture, sequencing is each shell's own (caution (x)).
# Usage: run_mixpow_kcal.sh <k> <gpu>
set -u
K=${1:?k}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
TAG=$(echo "$K" | tr -d '.')
LOG=output/logs/mpkcal_$TAG.log
rm -f ~/v/logs/mpkcal_$TAG.done ~/v/logs/mpkcal_$TAG.fail

echo "[kcal:$K] START $(date +%H:%M:%S) gpu=$GPU" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values "$K" --trajectories-per-prompt 1 \
  --cap-factual 200 --cap-neutral 0 --cap-creative 0 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --data-dir data/bench/alpaca --max-new-tokens 200 --batch-size 64 \
  --output-dir "output/mixpow/kcal_$TAG" >> "$LOG" 2>&1
RC=$?
echo "[kcal:$K] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/mpkcal_$TAG.done || touch ~/v/logs/mpkcal_$TAG.fail
