#!/usr/bin/env bash
# results/onset_prediction_comma7b_seed8.md (feat-133) -- one prompt class of the Comma-7B seed
# replication, CORRECTED PROTOCOL.
#
# EXACTLY ONE THING CHANGES from the arm on record: --seeds 42 43 44 -> 52 53 54.
# build_trajectory_seeds hashes the tuple into the high 16 bits, so the 64 trajectory seeds are
# disjoint from the original 64 by construction.
#
# --batch-size IS DELIBERATELY NOT PASSED. h1.py's default is 8, which is what
# output/phase5/sel_comma7b_64 used -- its launcher passed no flag either, so the default IS the
# protocol. feat-132 passed 32 and asserted in advance that batch size "does not change what is
# being drawn from"; its own integrity gate falsified that, with the draw-0 empty rate moving
# 45/200 -> 24/200 in the neutral class alone (z = 2.78). Caution (u): batch size is part of the
# seed, and at a RATE-valued quantity it is a shift rather than merely a re-roll. Do not add a
# --batch-size flag to this script.
#
# Splitting by prompt class does not change the batch size within a class, so it is not a second
# re-roll; feat-129 validated the split empirically (32,000 rewards bit-identical at ranks 0-63).
#
# Usage: run_comma7bseed8_gen.sh <gpu> <class>       class in {neutral,creative,factual}
set -u
GPU=$1; CLASS=$2
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
case "$CLASS" in
  neutral)  CAPS="--cap-neutral 200 --cap-creative 0 --cap-factual 0" ;;
  creative) CAPS="--cap-neutral 0 --cap-creative 150 --cap-factual 0" ;;
  factual)  CAPS="--cap-neutral 0 --cap-creative 0 --cap-factual 150" ;;
  *) echo "unknown class $CLASS" >&2; exit 2 ;;
esac
GEN=output/phase5/sel_comma7b_64_seed52_b8_$CLASS
LOG=output/logs/comma7bseed8_$CLASS.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[c7b8:$CLASS] $(date +%H:%M:%S) START $CLASS x 64 on GPU $GPU, batch = h1.py default" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 \
  --safe-model-path common-pile/comma-v0.1-2t \
  --risky-model-path common-pile/comma-v0.1-2t \
  --trajectories-per-prompt 64 --seeds 52 53 54 \
  $CAPS --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 \
  --output-dir "$GEN" >> "$LOG" 2>&1
RC=$?
echo "[c7b8:$CLASS] $(date +%H:%M:%S) generation rc=$RC" >> "$LOG"
# The marker is written ONLY on rc=0, so a failed class leaves the merger waiting rather than
# scoring a partial arm (caution (c): wait on a file, never on the absence of a pattern match).
[ $RC -eq 0 ] && date +%s > "$GEN/GEN_DONE"
exit $RC
