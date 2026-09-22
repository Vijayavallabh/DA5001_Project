#!/usr/bin/env bash
# feat-175: one more fixed opponent on the judged head-to-head. Generic over the opponent model;
# scripts/run_second_opponent.sh is the feat-153 instance of this and is kept verbatim so the
# committed opp2 arm's producing command is not rewritten under it.
#
# Bands: results/onset_prediction_opponent_ladder.md.
#
# ONLY --baseline-dir CHANGES. The selection arm, the metered arm, both controls, the reward cache,
# the judge, the seed and the prompt set are the committed ones, byte for byte.
#
# Usage: run_opponent.sh <model-id> <tag> <cards> [--device-map auto]
set -uo pipefail
MODEL=${1:?model}; TAG=${2:?tag}; CARDS=${3:?cards}; DM=${4:-}
source "$HOME/v/env.sh"
cd "$HOME/v/DA5001_Project"
MARK="$HOME/v/logs/opp_$TAG"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CARDS"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
OUT=output/opponent_$TAG
{
echo "[opp:$TAG] model=$MODEL cards=$CARDS generate $(date +%H:%M)"
.venv/bin/python analysis/blocklist_decode.py \
  --model "$MODEL" --arms plain \
  --split ordinary --ngram 10 --chat --max-new 200 --temperature 1.0 --seed 1234 \
  --out "$OUT"
echo "[opp:$TAG] generate exit=$?"
echo "[opp:$TAG] judge $(date +%H:%M)"
.venv/bin/python analysis/order_averaged_h2h.py $DM \
  --sel-dir output/xfer/sel_anchor64 --metered-dir output/xfer/conc_all \
  --anchor-dir output/sweep_plain --baseline-dir "$OUT" \
  --rewards results/selection_rewards64.csv --n 64 --k 10 --seed 7717 \
  --tag "_opp_$TAG" --out results
echo "[opp:$TAG] judge exit=$?"
} > "${MARK}.log" 2>&1
if [ -f "results/order_averaged_h2h__opp_${TAG}.csv" ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
