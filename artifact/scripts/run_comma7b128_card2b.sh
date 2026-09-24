#!/usr/bin/env bash
# results/onset_prediction_comma7b_n128.md (feat-134), card 2 of 3: CREATIVE only, then the merge
# and the whole scoring pass to n=128.
#
# Replaces run_comma7b128_card2.sh, which did creative+factual on one card under the registered
# two-card plan. GPU 1 came free minutes after launch, so factual moved to card 3 and the wall clock
# drops from ~19 h to ~12 h. A NEW launcher rather than an edit to the running one, per the
# multi-GPU rule. Nothing measured changes; the scoring log records the actual allocation.
#
# --batch-size IS DELIBERATELY NOT PASSED -- see card 1. Do not add the flag.
#
# Usage: run_comma7b128_card2b.sh [gpu]
set -u
GPU=${1:-4}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/comma7b128_card2.log
GEN=output/phase5/sel_comma7b_128_creative
NEU=output/phase5/sel_comma7b_128_neutral
FAC=output/phase5/sel_comma7b_128_factual
MERGED=output/phase5/sel_comma7b_128
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[c2b] $(date +%H:%M:%S) START creative 150 x 128 on GPU $GPU, batch = h1.py default" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 \
  --safe-model-path common-pile/comma-v0.1-2t \
  --risky-model-path common-pile/comma-v0.1-2t \
  --trajectories-per-prompt 128 --seeds 42 43 44 \
  --cap-neutral 0 --cap-creative 150 --cap-factual 0 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 \
  --output-dir "$GEN" >> "$LOG" 2>&1
RC=$?
echo "[c2b] $(date +%H:%M:%S) generation rc=$RC" >> "$LOG"
if [ $RC -ne 0 ]; then echo "[c2b] ABORT: own generation failed" >> "$LOG"; exit 1; fi

# Wait on FILES the other two cards write only on rc=0 (caution (c)), with the trace on its own
# descriptor so compute_hours.py's traced_sleep() subtracts the waiting instead of billing it.
exec 9>>"$LOG"
BASH_XTRACEFD=9
set -x
WAITED=0
for D in "$NEU" "$FAC"; do
  while [ ! -f "$D/GEN_DONE" ]; do
    sleep 60; WAITED=$((WAITED + 60))
    if [ $WAITED -gt 43200 ]; then
      echo "[c2b] ABORT: still waiting after 12h" >> "$LOG"; exit 1
    fi
  done
done
set +x
echo "[c2b] $(date +%H:%M:%S) all three classes done after ${WAITED}s of waiting; merging" >> "$LOG"

mkdir -p "$MERGED"
cp "$NEU/trajectories_k0_neutral.jsonl" "$MERGED/" || exit 1
cp "$GEN/trajectories_k0_creative.jsonl" "$MERGED/" || exit 1
cp "$FAC/trajectories_k0_factual.jsonl" "$MERGED/" || exit 1
wc -l "$MERGED"/*.jsonl >> "$LOG"

echo "[c2b] $(date +%H:%M:%S) START scoring and judging to n=128" >> "$LOG"
env $E .venv/bin/python analysis/selection_scaling.py \
  --gen-dir "$MERGED" \
  --reward-cache results/selection_rewards128_comma7b.csv \
  --max-n 128 \
  --tag _comma7b128 --out results >> "$LOG" 2>&1
echo "[c2b] $(date +%H:%M:%S) scoring rc=$? -- COMMA-7B n=128 ARM DRAINED" >> "$LOG"
