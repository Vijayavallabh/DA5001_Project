#!/usr/bin/env bash
# results/onset_prediction_tokenswap_aux.md (feat-167), Half A: vet each candidate auxiliary with
# the protocol Appendix D already uses for anchors -- the model ALONE, no defence, 100-token raw
# prefix, near-verbatim recall on the 100 protected passages.
# Usage: run_tokenswap_aux.sh <tag> <model> <gpu>
set -u
TAG=${1:?tag}; MODEL=${2:?model}; GPU=${3:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/tsaux_$TAG.log
rm -f ~/v/logs/tsaux_$TAG.done ~/v/logs/tsaux_$TAG.fail

echo "[aux:$TAG] START $(date +%H:%M:%S) gpu=$GPU model=$MODEL" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/blocklist_decode.py --model "$MODEL" \
  --split attack_train --limit 100 --seed-tokens 100 --raw-prompt --ngram 10 \
  --arms plain --max-new 200 --temperature 1.0 --seed 1234 \
  --out "output/tsaux/vet_$TAG" >> "$LOG" 2>&1
RC=$?
echo "[aux:$TAG] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/tsaux_$TAG.done || touch ~/v/logs/tsaux_$TAG.fail
