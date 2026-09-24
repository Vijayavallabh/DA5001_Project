#!/usr/bin/env bash
# feat-178: one rung of the opponent ladder ON ALPACAEVAL. Bands:
# results/onset_prediction_opponent_by_workload.md.
#
# ONLY --baseline-dir differs from scripts/run_mixpow_judge_k.sh, which produced the committed
# AlpacaEval binding-budget reading. The selection arm, the metered arm (output/mixpow/conc_k10
# holds k=1 -- that tree's directory names are swapped with respect to their contents), the anchor
# control, the 51,520 reward scores, the judge, the seed and the prompt set are that pass's, byte
# for byte.
#
# EVERY rung goes through blocklist_decode, including the Llama one that the committed opponent
# directory was generated differently for, so this ladder's slope is measured inside one pipeline
# (caution (at): two arms compared must come from the same pipeline).
#
# Usage: run_oppalp.sh <model-id> <tag> <gpu>
set -uo pipefail
MODEL=${1:?model}; TAG=${2:?tag}; GPU=${3:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
MARK="$HOME/v/logs/oppalp_$TAG"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
OUT=output/oppalp_$TAG
LOG=output/logs/oppalp_$TAG.log
{
# A MODEL DIRECTORY IS NOT A MODEL: an empty 24K skeleton lists identically to a 28G checkpoint.
D="hf_cache/models--${MODEL//\//--}"
if [ ! -s "$D/refs/main" ] || ! ls "$D"/snapshots/*/*.safetensors >/dev/null 2>&1; then
  echo "[oppalp:$TAG] PREFLIGHT FAIL: $MODEL is not resolvable offline under $D"; exit 3
fi
echo "[oppalp:$TAG] model=$MODEL gpu=$GPU generate $(date +%H:%M:%S)"
.venv/bin/python analysis/blocklist_decode.py \
  --model "$MODEL" --arms plain --data-dir data/bench/alpaca \
  --split ordinary --ngram 10 --chat --max-new 200 --temperature 1.0 --seed 1234 \
  --out "$OUT"
GRC=$?
echo "[oppalp:$TAG] generate exit=$GRC $(date +%H:%M:%S)"
# A FAILED GENERATION MUST NOT REACH THE JUDGE: a PARTIAL directory judges a quietly smaller
# prompt set and looks entirely healthy.
[ $GRC -eq 0 ] || { echo "[oppalp:$TAG] ABORT: generation failed, not judging"; exit $GRC; }
echo "[oppalp:$TAG] judge $(date +%H:%M:%S)"
.venv/bin/python analysis/order_averaged_h2h.py --judge microsoft/Phi-3.5-mini-instruct \
  --sel-dir output/mixpow/sel_anchor64 --metered-dir output/mixpow/conc_k10 \
  --anchor-dir output/mixpow/sel_anchor64 --baseline-dir "$OUT" \
  --rewards results/mixpow_rewards64.csv --data-dir data/bench/alpaca \
  --tag "_oppalp_$TAG" --seed 7717 --n 64 --k 1 --out results
echo "[oppalp:$TAG] judge exit=$? $(date +%H:%M:%S)"
} > "$LOG" 2>&1
if [ -f "results/order_averaged_h2h__oppalp_${TAG}.csv" ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
