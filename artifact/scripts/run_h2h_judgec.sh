#!/usr/bin/env bash
# Arm A of results/onset_prediction_h2h_independent.md: the SAME generations, judged by judge C
# (Meta-Llama-3.1-8B-Instruct) instead of judge B. Judge C is the fixed opponent's own checkpoint,
# so self-preference runs against the hypothesis under test -- the harsh direction, chosen on purpose.
# Usage: run_h2h_judgec.sh <gpu>
set -u
GPU=${1:-2}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/h2h_judgec.log
echo "[h2hC] start $(date +%H:%M) on GPU $GPU" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_averaged_h2h.py \
    --judge meta-llama/Meta-Llama-3.1-8B-Instruct --out results >> "$LOG" 2>&1
echo "[h2hC] exit=$? at $(date +%H:%M)" >> "$LOG"
for f in results/order_averaged_h2h.csv results/order_averaged_h2h_per_prompt.csv; do
  [ -f "$f" ] && cp "$f" "${f%.csv}_seed42_judgeC.csv"
done
# restore the canonical file: this arm must not move the committed original
cp results/order_averaged_h2h_seed42_judgeB.csv results/order_averaged_h2h.csv
cp results/order_averaged_h2h_per_prompt_seed42_judgeB.csv results/order_averaged_h2h_per_prompt.csv
echo "[h2hC] restored canonical CSV; DONE $(date +%H:%M)" >> "$LOG"
