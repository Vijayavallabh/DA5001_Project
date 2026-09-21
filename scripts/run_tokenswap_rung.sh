#!/usr/bin/env bash
# results/onset_prediction_tokenswap_aux.md (feat-167), Half B: the utility rung at one auxiliary.
# Only the swap arm is generated -- the rule-off control does not depend on the auxiliary and
# feat-165 generates it once, so every rung is compared against the same control.
# Flags match scripts/run_memfree.sh (caution (at)).
# Usage: run_tokenswap_rung.sh <tag> <aux-model> <gpu>
set -u
TAG=${1:?tag}; AUX=${2:?aux}; GPU=${3:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/tsrung_$TAG.log
rm -f ~/v/logs/tsrung_$TAG.done ~/v/logs/tsrung_$TAG.fail

echo "[rung:$TAG] START $(date +%H:%M:%S) gpu=$GPU aux=$AUX" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/tokenswap_decode.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct --aux "$AUX" \
  --split ordinary --chat --max-new 200 --temperature 1.0 --seed 1234 \
  --arms tokenswap --out "output/tokenswap/util_$TAG" >> "$LOG" 2>&1
RC=$?
echo "[rung:$TAG] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/tsrung_$TAG.done || touch ~/v/logs/tsrung_$TAG.fail
