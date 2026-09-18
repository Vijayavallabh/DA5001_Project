#!/usr/bin/env bash
# results/onset_prediction_n256.md, card 2 of 2: the CREATIVE+FACTUAL half of Arm A's generation,
# then the merge and the whole scoring pass.
#
# One queue shell, one card, jobs in series, no PID captured (caution (x)). The wait for card 1 is
# on a FILE card 1 writes only on rc=0, never on a process or a pattern (caution (c), which has cost
# eight incidents including one waiter that could never exit).
#
# Usage: run_n128_card2.sh [gpu]
set -u
GPU=${1:-2}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/n128_card2.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"
GEN=output/phase5/sel_anchor128_cf
NEU=output/phase5/sel_anchor128_neutral
MERGED=output/phase5/sel_anchor128

echo "[c2] $(date +%H:%M:%S) START Arm A generation, creative 150 + factual 150 x 128, GPU $GPU" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 128 \
  --seeds 42 43 44 \
  --cap-neutral 0 --cap-creative 150 --cap-factual 150 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 --batch-size 64 \
  --output-dir "$GEN" >> "$LOG" 2>&1
RC=$?
echo "[c2] $(date +%H:%M:%S) generation rc=$RC" >> "$LOG"
if [ $RC -ne 0 ]; then echo "[c2] ABORT: own generation failed" >> "$LOG"; exit 1; fi

echo "[c2] $(date +%H:%M:%S) waiting for card 1's $NEU/GEN_DONE" >> "$LOG"
WAITED=0
while [ ! -f "$NEU/GEN_DONE" ]; do
  sleep 60; WAITED=$((WAITED + 60))
  if [ $WAITED -gt 43200 ]; then
    echo "[c2] ABORT: card 1 has not finished after 12h" >> "$LOG"; exit 1
  fi
done
echo "[c2] $(date +%H:%M:%S) card 1 done after ${WAITED}s of waiting; merging" >> "$LOG"

# The three per-class files never collide, so the merge is three copies. load_candidates() globs
# trajectories_k0_<class>.jsonl and skips what is absent (analysis/selection_decoding.py:56).
mkdir -p "$MERGED"
cp "$NEU/trajectories_k0_neutral.jsonl" "$MERGED/" || exit 1
cp "$GEN/trajectories_k0_creative.jsonl" "$GEN/trajectories_k0_factual.jsonl" "$MERGED/" || exit 1
wc -l "$MERGED"/*.jsonl >> "$LOG"

echo "[c2] $(date +%H:%M:%S) START scoring and judging to n=128" >> "$LOG"
env $E .venv/bin/python analysis/selection_scaling.py \
  --gen-dir "$MERGED" \
  --reward-cache results/selection_rewards128.csv \
  --max-n 128 \
  --tag _n128 --out results >> "$LOG" 2>&1
echo "[c2] $(date +%H:%M:%S) scoring rc=$? -- CARD 2 DRAINED" >> "$LOG"
