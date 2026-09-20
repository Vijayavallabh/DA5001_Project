#!/usr/bin/env bash
# feat-157: CoTaEval news with a LARGER SCORER. Bands: results/onset_prediction_cotaeval_scorer.md.
# Only --reward-model changes from feat-155. Usage: <gpu> <anchor> <tag> [seed]
set -u
GPU=$1; MODEL=$2; TAG=$3; SEED=${4:-8801}
cd "$(dirname "$0")/.."
mkdir -p output/logs output/phase5/verifiable
LOG="output/logs/${TAG}.log"
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"
echo "[$TAG] $(date +%H:%M:%S) START $MODEL scorer=Qwen2.5-14B seed=$SEED gpu=$GPU" >> "$LOG"
env $E .venv/bin/python analysis/selection_verifiable.py --task cotaeval \
  --anchor "$MODEL" --reward-model Qwen/Qwen2.5-14B-Instruct \
  --limit 500 --max-n 64 --max-new 24 --batch-size 16 --reward-batch-size 8 --seed "$SEED" \
  --gen-dir "output/phase5/$TAG" --tag "_$TAG" --out results >> "$LOG" 2>&1
RC=$?
echo "[$TAG] $(date +%H:%M:%S) exit=$RC" >> "$LOG"
[ $RC -eq 0 ] && touch "$HOME/v/logs/${TAG}.done" || touch "$HOME/v/logs/${TAG}.fail"
