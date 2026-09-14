#!/usr/bin/env bash
# feat-103: the corrected natural-memorisation arm -- raw passage seed, no instruction header.
set -u
LOG=output/logs/extraction_70b_raw.log
mkdir -p output/logs
set -a; . ./.env; set +a
echo "[e70raw] start $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py \
    --risky-model unsloth/Meta-Llama-3.1-70B --raw-prompt \
    --risky-device-map auto --max-memory 0=75GiB,1=75GiB \
    --n-values 1 8 64 --limit 100 --batch-size 8 \
    --prefix selection_extraction_70b_raw --out results >> "$LOG" 2>&1
echo "[e70raw] exit=$? at $(date +%H:%M)" >> "$LOG"
