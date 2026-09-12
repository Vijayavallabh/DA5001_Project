#!/usr/bin/env bash
# feat-102 entry gate: is BookMIA-seen extractable by an 8B model at all?
# The paper's leakage claim rests on 100 CopyBench passages and ONE LoRA memoriser. Scaling it to
# 50 real books needs a risky model that naturally memorised them -- and whether Llama-3.1-8B did
# is an empirical question its own k=-1 baseline answers. Caution (a)'s threshold: sampled recall
# >= 0.10 or the corpus carries nothing to extract and the arm is uninformative.
set -u
WAIT_PID="${1:-}"
LOG=output/logs/bookmia_extraction_gate.log
mkdir -p output/logs
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  sleep 45
fi
set -a; . ./.env; set +a
echo "[gate] start $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py \
    --data data/bench/bookmia100 --split test \
    --risky-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --n-values 1 --limit 200 --prefix bookmia_extraction_gate \
    --out results >> "$LOG" 2>&1
echo "[gate] exit=$? at $(date +%H:%M)" >> "$LOG"
