#!/usr/bin/env bash
# feat-108: the two mechanisms' leakage on one set of passages with one control.
# Bands: results/onset_prediction_leakage_headtohead.md (G0 gate first, then M1-M3).
# Usage: run_leakage_headtohead.sh <gpu> [wait-pid]
set -u
GPU=$1; WAIT_PID="${2:-}"
LOG=output/logs/leakage_headtohead.log
mkdir -p output/logs results/h2h output/h2h output/h2h_figures
if [ -n "$WAIT_PID" ]; then
  echo "[l2l] waiting on PID $WAIT_PID at $(date +%H:%M)" >> "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  sleep 30
fi
set -a; . ./.env; set +a
ENVV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

# Arm A, metered. Only `single` is comparable to Arm B: oracle and chained are many-query
# composition attacks with no counterpart there, and the pre-registration excludes them.
echo "[l2l] metered $(date +%H:%M) on GPU ${GPU}" >> "$LOG"
env $ENVV .venv/bin/python analysis/composition_attack.py \
    --risky-model output/memorizing_llama8b --modes single --windows 20 \
    --k-values -1 0 0.5 1 3 20 --limit 100 --batch-size 32 \
    --out results/h2h --figures output/h2h_figures \
    --text-out output/h2h/extracted.csv --queries-out output/h2h/queries.jsonl >> "$LOG" 2>&1
echo "[l2l] metered exit=$? at $(date +%H:%M)" >> "$LOG"

# Arm B, selection, at the generation length Arm A gets. --batch-size stays at the default 32:
# The prefix deliberately does NOT start with selection_extraction: tests/test_selection_claims.py
# counts results/selection_extraction*.csv as the per-ANCHOR leakage arms, and this is the same
# anchor at a different generation length, not a fifth anchor.
# the k=-1 arm is sampled and the batch size is part of the seed (AGENTS.md caution (u)).
echo "[l2l] selection $(date +%H:%M)" >> "$LOG"
env $ENVV .venv/bin/python analysis/selection_extraction.py \
    --risky-model output/memorizing_llama8b --n-values 1 8 64 --limit 100 \
    --max-new-tokens 300 --prefix leakage_len300 --out results >> "$LOG" 2>&1
echo "[l2l] selection exit=$? at $(date +%H:%M)" >> "$LOG"
