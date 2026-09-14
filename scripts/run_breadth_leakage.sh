#!/usr/bin/env bash
# feat-107 (C4): near-verbatim recall at a new breadth anchor, under the adversarial selector.
# BATCH SIZE IS PART OF THE SEED. The k=-1 memoriser baseline is a SAMPLED arm, and generate()
# consumes the RNG per chunk, so --batch-size 64 produced a different draw: the first three
# anchors agree with each other on 100/100 passages and the batch-64 run agreed on 24/100,
# moving the baseline 0.3925 -> 0.4434. Leave this at the default 32 that every anchor on record
# used, or tests/test_selection_claims.py stops being able to detect the bug it was written for.
# Queued behind that anchor's generation on the same card.
# Usage: run_breadth_leakage.sh <gpu> <model-id> <tag> <wait-pid>
set -u
GPU=$1; MODEL=$2; TAG=$3; WAIT_PID=${4:-}
LOG=output/logs/leakage${TAG}.log
mkdir -p output/logs
if [ -n "$WAIT_PID" ]; then
  echo "[lk${TAG}] waiting on PID $WAIT_PID at $(date +%H:%M)" >> "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  sleep 30
fi
set -a; . ./.env; set +a
echo "[lk${TAG}] start $(date +%H:%M) on GPU ${GPU}" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py \
    --safe-model "$MODEL" --risky-model output/memorizing_llama8b \
    --n-values 1 8 64 --limit 100 --batch-size 32 \
    --prefix "selection_extraction${TAG}" --out results >> "$LOG" 2>&1
echo "[lk${TAG}] exit=$? at $(date +%H:%M)" >> "$LOG"
