#!/usr/bin/env bash
# results/onset_prediction_kl3m_seed.md -- one prompt class of the KL3M-1.7B seed replication.
#
# ONE THING CHANGES from the arm on record: --seeds 42 43 44 -> 52 53 54. build_trajectory_seeds
# hashes the tuple into the high 16 bits, so the 64 trajectory seeds are disjoint from the original
# 64 by construction. Everything else is held: --batch-size 32 (caution (u)), 200 max-new-tokens,
# the same 500 prompts, --trajectories-per-prompt 64.
#
# Splitting by prompt class is not merely argued to be safe: feat-129 split the same way and its
# merged pool reproduced the single-card pool's 32,000 rewards BIT-IDENTICALLY at ranks 0-63.
#
# Usage: run_kl3mseed_gen.sh <gpu> <class>       class in {neutral,creative,factual}
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
GEN=output/phase5/sel_kl3m17b_64_seed52_$CLASS
LOG=output/logs/kl3mseed_$CLASS.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[seed:$CLASS] $(date +%H:%M:%S) START $CLASS x 64 on GPU $GPU" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 \
  --safe-model-path alea-institute/kl3m-003-1.7b \
  --risky-model-path alea-institute/kl3m-003-1.7b \
  --trajectories-per-prompt 64 --seeds 52 53 54 \
  $CAPS --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 --batch-size 32 \
  --output-dir "$GEN" >> "$LOG" 2>&1
RC=$?
echo "[seed:$CLASS] $(date +%H:%M:%S) generation rc=$RC" >> "$LOG"
# The marker is written ONLY on rc=0, so a failed class leaves the merger waiting rather than
# scoring a partial arm (caution (c): wait on a file, never on the absence of a pattern match).
[ $RC -eq 0 ] && date +%s > "$GEN/GEN_DONE"
exit $RC
