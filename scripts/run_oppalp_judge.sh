#!/usr/bin/env bash
# feat-178: re-run ONLY the judge step of scripts/run_oppalp.sh, for rungs whose generation finished
# (`generate exit=0`) and whose judge died of a mechanical fault. On 2026-09-23 the judges of
# qwen05b, qwen15b and qwen3b hit CUDA out-of-memory on cards the sibling project's vLLM workers hold
# at 70-75 GB; their generations are untouched. The command below is run_oppalp.sh's judge step
# character for character, so this is a re-run at the identical specification, which is what a
# mechanical failure calls for. It writes the same .done/.fail marker the original would have.
# ONE QUEUE SHELL, JOBS IN ORDER (caution (x)). Usage: run_oppalp_judge.sh <gpu> <tag> [<tag> ...]
set -uo pipefail
GPU=${1:?gpu}; shift
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
for TAG in "$@"; do
  MARK="$HOME/v/logs/oppalp_$TAG"
  LOG=output/logs/oppalp_${TAG}_rejudge.log
  if ! grep -q "generate exit=0" "output/logs/oppalp_$TAG.log"; then
    echo "[oppalp:$TAG] generation did not finish; not judging" > "$LOG"; continue
  fi
  rm -f "${MARK}.done" "${MARK}.fail"
  {
  echo "[oppalp:$TAG] re-judge on GPU $GPU $(date +%H:%M:%S)"
  .venv/bin/python analysis/order_averaged_h2h.py --judge microsoft/Phi-3.5-mini-instruct \
    --sel-dir output/mixpow/sel_anchor64 --metered-dir output/mixpow/conc_k10 \
    --anchor-dir output/mixpow/sel_anchor64 --baseline-dir "output/oppalp_$TAG" \
    --rewards results/mixpow_rewards64.csv --data-dir data/bench/alpaca \
    --tag "_oppalp_$TAG" --seed 7717 --n 64 --k 1 --out results
  echo "[oppalp:$TAG] judge exit=$? $(date +%H:%M:%S)"
  } > "$LOG" 2>&1
  if [ -f "results/order_averaged_h2h__oppalp_${TAG}.csv" ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
done
