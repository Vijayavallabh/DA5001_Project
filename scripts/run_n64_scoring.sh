#!/usr/bin/env bash
# feat-100 scoring: the strongest anchor at the largest n. GPU 4.
set -u
LOG=output/logs/sel_comma7b_64_scoring.log
mkdir -p output/logs
set -a; . ./.env; set +a
echo "[n64] start $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_scaling.py --gen-dir output/phase5/sel_comma7b_64 \
    --max-n 64 --reward-cache results/selection_rewards64_comma7b.csv \
    --tag _comma7b64 --out results >> "$LOG" 2>&1
echo "[n64] exit=$? at $(date +%H:%M)" >> "$LOG"
