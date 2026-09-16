#!/usr/bin/env bash
# Same arm as run_vetting_protocol.sh, split across cards after the user re-opened multi-GPU use
# at 19:25. The protocol is byte-identical to the registered one -- only WHICH CARD runs a model
# changes, and the card is not a registered parameter. Do not alter a flag here: they are all
# committed in results/onset_prediction_vetting_protocol.md.
#
# A NEW FILE rather than an edit to run_vetting_protocol.sh, which a live shell was reading:
# bash reads a script incrementally by byte offset, and editing one mid-run produced a spurious
# FAILED on 2026-09-16 after all the work had completed.
#
# One queue shell per card, jobs in order, no PID captured by guesswork (caution (x)): if this
# card must wait for work already running, its PID is passed in explicitly and polled with kill -0.
# Usage: run_vetting_split.sh <gpu> <wait-pid-or-dash> <tag:model> [<tag:model> ...]
set -u
GPU=$1; shift
WAIT_PID=$1; shift
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a

if [ "$WAIT_PID" != "-" ]; then
  echo "[vetsplit:gpu$GPU] waiting on PID $WAIT_PID at $(date +%H:%M)" >> output/logs/vet_queue.log
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  sleep 20
fi

for SPEC in "$@"; do
  TAG=${SPEC%%:*}; MODEL=${SPEC#*:}
  LOG="output/logs/vet_${TAG}.log"
  echo "[vet:${TAG}] start $(date +%H:%M) on GPU ${GPU} model=${MODEL}" >> "$LOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_extraction.py \
      --safe-model "$MODEL" --risky-model output/memorizing_llama8b \
      --raw-prompt --split test --novel harry_potter --limit 50 \
      --seed-tokens 100 --max-new-tokens 200 \
      --n-values 1 8 64 --batch-size 8 \
      --prefix "vet_${TAG}" --out results >> "$LOG" 2>&1
  echo "[vet:${TAG}] exit=$? at $(date +%H:%M)" >> "$LOG"
done
echo "[vetsplit:gpu$GPU] drained at $(date +%H:%M)" >> output/logs/vet_queue.log
