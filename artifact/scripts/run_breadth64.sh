#!/usr/bin/env bash
# results/onset_prediction_breadth64.md: take ONE breadth anchor from n<=8 to n<=64.
#
# Every part of the seed is held at the breadth arm's value so that draws 0-7 of the 64-draw pool
# are bit-identical to the committed 8-draw pool and the n<=8 half of the sweep reproduces
# results/selection_scaling_<name>.csv:
#   * seeds 42 43 44          -- the dap/e1.py:468 default, which run_breadth_anchor.sh took implicitly
#   * --batch-size 32         -- the breadth arm's value, NOT sel_anchor64's 64 (caution (u))
#   * --max-new-tokens 200, --cap-neutral 200 --cap-creative 150 --cap-factual 150 (the same 500 prompts)
# E1 groups jobs by seed before batching (dap/e1.py:325) and seeds each generate() from that group's
# own value (dap/e1.py:307), so a seed group is the same 500 jobs in the same buckets either way.
#
# Usage: run_breadth64.sh <gpu> <model-id> <name>
set -u
GPU=$1; MODEL=$2; NAME=$3
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
GEN=output/phase5/sel_${NAME}_64
TAG=_${NAME}64
CACHE=results/selection_rewards64_${NAME}.csv
LOG=output/logs/breadth64_${NAME}.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[b64:$NAME] $(date +%H:%M:%S) START generation, 500 x 64, $MODEL on GPU $GPU" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 \
  --safe-model-path "$MODEL" --risky-model-path "$MODEL" \
  --trajectories-per-prompt 64 \
  --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 --batch-size 32 \
  --output-dir "$GEN" >> "$LOG" 2>&1
RC=$?
echo "[b64:$NAME] $(date +%H:%M:%S) generation rc=$RC" >> "$LOG"
[ $RC -ne 0 ] && { echo "[b64:$NAME] generation failed -- NOT scoring half an arm" >> "$LOG"; exit $RC; }

echo "[b64:$NAME] $(date +%H:%M:%S) START scoring, max-n 64, tag $TAG" >> "$LOG"
env $E .venv/bin/python analysis/selection_scaling.py --gen-dir "$GEN" \
  --max-n 64 --reward-cache "$CACHE" --tag "$TAG" --out results >> "$LOG" 2>&1
echo "[b64:$NAME] $(date +%H:%M:%S) scoring rc=$? -- $NAME DONE" >> "$LOG"
