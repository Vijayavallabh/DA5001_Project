#!/usr/bin/env bash
# feat-107: one more anchor for the six-anchor breadth arm.
# Usage: run_breadth_anchor.sh <gpu> <model-id> <tag>
set -u
GPU=$1; MODEL=$2; TAG=$3
LOG=output/logs/breadth${TAG}.log
mkdir -p output/logs
set -a; . ./.env; set +a
echo "[br${TAG}] generate $(date +%H:%M) on GPU ${GPU}" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0.0 \
    --safe-model-path "$MODEL" --risky-model-path "$MODEL" \
    --trajectories-per-prompt 8 --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
    --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 --batch-size 32 \
    --output-dir "output/phase5/sel${TAG}_8" >> "$LOG" 2>&1
echo "[br${TAG}] generate exit=$? at $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_scaling.py --gen-dir "output/phase5/sel${TAG}_8" \
    --max-n 8 --reward-cache "results/selection_rewards8${TAG}.csv" \
    --tag "$TAG" --out results >> "$LOG" 2>&1
echo "[br${TAG}] score exit=$? at $(date +%H:%M)" >> "$LOG"
