#!/usr/bin/env bash
# feat-102: hand the adversarial selector to a memoriser we did not make.
# GPUs 1+2 (the 70B is 141 GB in bf16 and needs two cards; caution (q): the split must be explicit).
set -u
WAIT_PID="${1:-}"
LOG=output/logs/extraction_70b.log
mkdir -p output/logs
if [ -n "$WAIT_PID" ]; then
  echo "[e70] waiting on PID $WAIT_PID at $(date +%H:%M)" >> "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  sleep 45
fi
set -a; . ./.env; set +a
echo "[e70] start $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py \
    --risky-model unsloth/Meta-Llama-3.1-70B \
    --risky-device-map auto --max-memory 0=75GiB,1=75GiB \
    --n-values 1 8 64 --limit 100 --batch-size 8 \
    --prefix selection_extraction_70b --out results >> "$LOG" 2>&1
echo "[e70] exit=$? at $(date +%H:%M)" >> "$LOG"
