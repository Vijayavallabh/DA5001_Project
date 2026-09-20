#!/usr/bin/env bash
# feat-156: one CoTaEval news arm on one card. Bands: results/onset_prediction_cotaeval_breadth.md.
# Usage: run_cotaeval_anchor.sh <gpu> <anchor-model-id> <tag> [seed]
# ONE shell per card, no PID is ever captured (caution (x)); batch size is HELD at 16 (caution (u)).
set -u
GPU=$1; MODEL=$2; TAG=$3; SEED=${4:-8801}
cd "$(dirname "$0")/.."
mkdir -p output/logs output/phase5/verifiable
LOG="output/logs/cta_${TAG}.log"
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"
echo "[cta:$TAG] $(date +%H:%M:%S) START $MODEL seed=$SEED on GPU $GPU" >> "$LOG"
env $E .venv/bin/python analysis/selection_verifiable.py --task cotaeval \
  --anchor "$MODEL" --limit 500 --max-n 64 --max-new 24 \
  --batch-size 16 --reward-batch-size 16 --seed "$SEED" \
  --gen-dir "output/phase5/cta_$TAG" \
  --tag "_$TAG" --out results >> "$LOG" 2>&1
RC=$?
echo "[cta:$TAG] $(date +%H:%M:%S) exit=$RC" >> "$LOG"
[ $RC -eq 0 ] && touch "$HOME/v/logs/cta_${TAG}.done" || touch "$HOME/v/logs/cta_${TAG}.fail"
