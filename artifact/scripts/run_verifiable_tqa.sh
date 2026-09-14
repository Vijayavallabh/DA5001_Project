#!/usr/bin/env bash
# feat-105: the judge-free axis on a knowledge task. GPU 4.
set -u
LOG=output/logs/verifiable_tqa.log
mkdir -p output/logs output/phase5/verifiable
set -a; . ./.env; set +a
echo "[tqa] start $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_verifiable.py --task triviaqa \
    --anchor common-pile/comma-v0.1-2t --limit 500 --max-n 64 --n-shot 5 --max-new 24 \
    --batch-size 32 --reward-batch-size 16 --tag _tqa_comma7b \
    --out results >> "$LOG" 2>&1
echo "[tqa] exit=$? at $(date +%H:%M)" >> "$LOG"
