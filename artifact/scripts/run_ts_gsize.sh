#!/usr/bin/env bash
# results/onset_prediction_tokenswap_gsize.md (feat-169): one rung of the shrunken-G ladder.
#
# The auxiliary is FIXED at DistilGPT-2 -- the one TokenSwap's own paper uses, and the one that
# suppresses completely at full G -- so that |G| is a cause this arm sets rather than a correlate
# it observes. Every other flag is feat-165's leakage half verbatim (caution (at)); only --g-words
# and --g-seed change.
#
# Usage: run_ts_gsize.sh <words> <g-seed> <gpu>
set -u
W=${1:?words}; GS=${2:?g-seed}; GPU=${3:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
MEM=output/memorizing_llama8b
BASE=meta-llama/Meta-Llama-3.1-8B-Instruct
AUX=distilbert/distilgpt2
TAG=w${W}s${GS}
LOG=output/logs/tsg_$TAG.log
rm -f ~/v/logs/tsg_$TAG.done ~/v/logs/tsg_$TAG.fail

echo "[gsize:$TAG] START $(date +%H:%M:%S) gpu=$GPU words=$W g-seed=$GS" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/tokenswap_decode.py \
  --model "$MEM" --base-model "$BASE" --aux "$AUX" \
  --split attack_train --limit 100 --seed-tokens 20 --max-new 200 --temperature 1.0 \
  --g-words "$W" --g-seed "$GS" \
  --arms tokenswap --out "output/tokenswap/gsize_$TAG" >> "$LOG" 2>&1
RC=$?
echo "[gsize:$TAG] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/tsg_$TAG.done || touch ~/v/logs/tsg_$TAG.fail
