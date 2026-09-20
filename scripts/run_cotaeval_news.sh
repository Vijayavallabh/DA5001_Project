#!/usr/bin/env bash
# feat-155: CoTaEval news, the standard-framework benchmark the PC report asked for.
# Bands: results/onset_prediction_cotaeval_news.md. Host B, one card per half, run in series by
# ONE queue shell so no PID is ever captured (caution (x)) and nothing waits on a pattern's
# absence (caution (c)).
set -u
GPU=${1:-0}
cd "$(dirname "$0")/.."
mkdir -p output/logs output/phase5/verifiable
LOG=output/logs/cotaeval_news.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[cta] $(date +%H:%M:%S) START utility 500 on GPU $GPU" >> "$LOG"
env $E .venv/bin/python analysis/selection_verifiable.py --task cotaeval \
  --anchor common-pile/comma-v0.1-2t --limit 500 --max-n 64 --max-new 24 \
  --batch-size 16 --reward-batch-size 16 --tag _cta_news --out results >> "$LOG" 2>&1
echo "[cta] $(date +%H:%M:%S) utility exit=$?" >> "$LOG"
touch "$HOME/v/logs/cotaeval.done"
echo "[cta] $(date +%H:%M:%S) CoTaEval ARM DRAINED" >> "$LOG"
