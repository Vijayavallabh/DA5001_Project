#!/usr/bin/env bash
# results/onset_prediction_workload_scope.md (feat-170) Arm C: the calibration sweep on OUR corpus.
# Reports activity rates only -- no judge, no utility, no band readable from it. The argmin rule
# and the grid are fixed in the registration, committed before this ran.
#
# LOCAL host. GPU is passed in; only 2 and 4 are free here (0 and 1 hold another user's 77 GB
# jobs, 3 is the 4 GB T400 and is never used). CUDA_DEVICE_ORDER=PCI_BUS_ID is mandatory: without
# it CUDA numbers the A100s 0-3 and the T400 4, so CUDA_VISIBLE_DEVICES=4 lands on the small card.
#
# Usage: run_wscope_kcal.sh <k> <gpu>
set -u
K=${1:?k}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
TAG=$(echo "$K" | tr -d '.')
LOG=output/logs/wskcal_$TAG.log
rm -f output/wscope/kcal_$TAG.done output/wscope/kcal_$TAG.fail
mkdir -p output/wscope

echo "[wskcal:$K] START $(date +%H:%M:%S) gpu=$GPU" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values "$K" --trajectories-per-prompt 1 \
  --cap-neutral 200 --cap-creative 0 --cap-factual 0 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --data-dir data --max-new-tokens 200 --batch-size 64 \
  --output-dir "output/wscope/kcal_$TAG" >> "$LOG" 2>&1
RC=$?
echo "[wskcal:$K] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch output/wscope/kcal_$TAG.done || touch output/wscope/kcal_$TAG.fail
