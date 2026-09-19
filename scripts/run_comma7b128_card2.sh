#!/usr/bin/env bash
# results/onset_prediction_comma7b_n128.md (feat-134), card 2 of 2: CREATIVE + FACTUAL, then the
# merge and the whole scoring pass to n=128.
#
# The longer generation goes here deliberately: this card also scores, so by the time it reaches the
# wait, card 1 is long finished and the wait is ~0. No idle hours are billed to anything.
#
# --batch-size IS DELIBERATELY NOT PASSED -- see card 1. Do not add the flag.
#
# Usage: run_comma7b128_card2.sh [gpu]
set -u
GPU=${1:-4}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/comma7b128_card2.log
GEN=output/phase5/sel_comma7b_128_cf
NEU=output/phase5/sel_comma7b_128_neutral
MERGED=output/phase5/sel_comma7b_128
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[c2] $(date +%H:%M:%S) START creative 150 + factual 150 x 128 on GPU $GPU, batch = default" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 \
  --safe-model-path common-pile/comma-v0.1-2t \
  --risky-model-path common-pile/comma-v0.1-2t \
  --trajectories-per-prompt 128 --seeds 42 43 44 \
  --cap-neutral 0 --cap-creative 150 --cap-factual 150 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 \
  --output-dir "$GEN" >> "$LOG" 2>&1
RC=$?
echo "[c2] $(date +%H:%M:%S) generation rc=$RC" >> "$LOG"
if [ $RC -ne 0 ]; then echo "[c2] ABORT: own generation failed" >> "$LOG"; exit 1; fi

# `set -x` through its own descriptor on the log so analysis/compute_hours.py's traced_sleep() can
# subtract the waiting; a bare `set -x` goes to stderr, which this launcher discards, and the
# odometer would then bill the wait as GPU time (2026-09-18: 7.69 idle hours across two merges).
exec 9>>"$LOG"
BASH_XTRACEFD=9
set -x
WAITED=0
while [ ! -f "$NEU/GEN_DONE" ]; do
  sleep 60; WAITED=$((WAITED + 60))
  if [ $WAITED -gt 43200 ]; then
    echo "[c2] ABORT: card 1 has not finished after 12h" >> "$LOG"; exit 1
  fi
done
set +x
echo "[c2] $(date +%H:%M:%S) card 1 done after ${WAITED}s of waiting; merging" >> "$LOG"

mkdir -p "$MERGED"
cp "$NEU/trajectories_k0_neutral.jsonl" "$MERGED/" || exit 1
cp "$GEN/trajectories_k0_creative.jsonl" "$GEN/trajectories_k0_factual.jsonl" "$MERGED/" || exit 1
wc -l "$MERGED"/*.jsonl >> "$LOG"

echo "[c2] $(date +%H:%M:%S) START scoring and judging to n=128" >> "$LOG"
env $E .venv/bin/python analysis/selection_scaling.py \
  --gen-dir "$MERGED" \
  --reward-cache results/selection_rewards128_comma7b.csv \
  --max-n 128 \
  --tag _comma7b128 --out results >> "$LOG" 2>&1
echo "[c2] $(date +%H:%M:%S) scoring rc=$? -- COMMA-7B n=128 ARM DRAINED" >> "$LOG"
