#!/usr/bin/env bash
# results/onset_prediction_comma7b_seed.md (feat-133) -- wait for all three classes, merge, score.
# Runs on GPU 1 after its own class finishes. Waits on FILES the generators write (caution (c)).
set -u
GPU=${1:-1}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
LOG=output/logs/comma7bseed8_merge.log
MERGED=output/phase5/sel_comma7b_64_seed52_b8
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

# `set -x` so analysis/compute_hours.py can SEE the sleeps. Its `traced_sleep()` subtracts every
# `+ sleep N` line a trace recorded, precisely so a shell that waits hours on a sentinel file is not
# billed as GPU time -- and without the trace it is. On 2026-09-18 that put 7.69 idle hours into the
# odometer across two merge shells (2.07 h and 5.62 h), which the total then carried as "at most".
# The number stayed a true upper bound; it was just 7.69 h looser than the work actually cost.
#
# BASH_XTRACEFD, not a bare `set -x`: xtrace goes to stderr, and these launchers are started with
# stderr discarded, so a bare `set -x` would write the trace nowhere the odometer can read it. Point
# the trace at its own descriptor on $LOG, and turn it off again after the wait so the rest of the
# script's output stays readable.
exec 9>>"$LOG"
BASH_XTRACEFD=9
set -x
WAITED=0
for C in neutral creative factual; do
  D=output/phase5/sel_comma7b_64_seed52_b8_$C
  echo "[merge] $(date +%H:%M:%S) waiting for $D/GEN_DONE" >> "$LOG"
  while [ ! -f "$D/GEN_DONE" ]; do
    sleep 60; WAITED=$((WAITED + 60))
    if [ $WAITED -gt 43200 ]; then
      echo "[merge] ABORT: still waiting after 12h" >> "$LOG"; exit 1
    fi
  done
done
set +x
echo "[merge] $(date +%H:%M:%S) all three classes done; merging" >> "$LOG"
mkdir -p "$MERGED"
for C in neutral creative factual; do
  cp "output/phase5/sel_comma7b_64_seed52_b8_$C/trajectories_k0_$C.jsonl" "$MERGED/" || exit 1
done
wc -l "$MERGED"/*.jsonl >> "$LOG"

echo "[merge] $(date +%H:%M:%S) START scoring, max-n 64, tag _comma7bseed52b8" >> "$LOG"
env $E .venv/bin/python analysis/selection_scaling.py --gen-dir "$MERGED" \
  --max-n 64 --reward-cache results/selection_rewards64_comma7bseed52b8.csv \
  --tag _comma7bseed52b8 --out results >> "$LOG" 2>&1
echo "[merge] $(date +%H:%M:%S) scoring rc=$? -- COMMA-7B BATCH-8 SEED ARM DRAINED" >> "$LOG"
