#!/usr/bin/env bash
# feat-101: the judge-free axis, queued behind the 70B imitation arm.
# Waits on a PID with `kill -0` -- never `pgrep -f`, which matches this shell (caution (c)).
set -u
WAIT_PID="${1:-}"
LOG=output/logs/verifiable_comma7b.log
mkdir -p output/logs output/phase5/verifiable
if [ -n "$WAIT_PID" ]; then
  echo "[verif] waiting on PID $WAIT_PID at $(date +%H:%M)" >> "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[verif] PID $WAIT_PID gone at $(date +%H:%M)" >> "$LOG"
  sleep 30   # let the CUDA child release its memory
fi
set -a; . ./.env; set +a
echo "[verif] start $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_verifiable.py \
    --anchor common-pile/comma-v0.1-2t --limit 500 --max-n 64 \
    --batch-size 32 --reward-batch-size 16 --tag _comma7b \
    --out results >> "$LOG" 2>&1
echo "[verif] exit=$? at $(date +%H:%M)" >> "$LOG"
