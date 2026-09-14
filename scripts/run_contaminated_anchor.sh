#!/usr/bin/env bash
# feat-111: what the log n certificate delivers when its premise fails.
# Every leakage arm on record has a clean anchor, so Pr_{p_s}[E] = 0 and Proposition 4's
# Pr_q[E] <= n Pr_{p_s}[E] is satisfied vacuously. These five anchors are LoRA-memorised copies
# already on disk, so the base rate is non-zero and the inequality is testable for the first time.
# Bands: results/onset_prediction_contaminated_anchor.md. Every band is read against THIS arm's
# own n=1 row -- the k=-1 numbers in output/phase5/fine_*/ are a different pipeline (caution (v)).
# Batch size stays at the default 32: the n=1 arm is sampled (caution (u)).
# Usage: run_contaminated_anchor.sh <gpu> <anchor-dir> <tag> [wait-pid]
set -u
GPU=$1; ANCHOR=$2; TAG=$3; WAIT_PID=${4:-}
LOG=output/logs/contam${TAG}.log
mkdir -p output/logs
if [ -n "$WAIT_PID" ]; then
  echo "[ct${TAG}] waiting on PID $WAIT_PID at $(date +%H:%M)" >> "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  sleep 30
fi
set -a; . ./.env; set +a
echo "[ct${TAG}] start $(date +%H:%M) on GPU ${GPU}, anchor ${ANCHOR}" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py \
    --safe-model "$ANCHOR" --risky-model output/memorizing_llama8b \
    --n-values 1 8 64 --limit 100 --batch-size 32 \
    --prefix "contam${TAG}" --out results >> "$LOG" 2>&1
echo "[ct${TAG}] exit=$? at $(date +%H:%M)" >> "$LOG"
