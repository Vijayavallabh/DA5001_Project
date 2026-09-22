#!/usr/bin/env bash
# results/onset_prediction_offsupport_ladder.md (feat-172) Arm B: the ON-support ladder control.
#
# The anchor on our own 850 ordinary prompts at n=256, sharded by prompt class so two cards work in
# parallel. Every flag matches feat-170 Arm A, whose n<=64 draws this extends -- --batch-size 64 in
# particular is part of the seed (caution (u)) and is what makes the bit-identity gate against
# results/wscope_rewards64_a.csv possible at all.
#
# LOCAL host: GPUs 2 and 4 are the free cards here (0 and 1 hold another user's jobs, 3 is the 4 GB
# T400 and is never used). CUDA_DEVICE_ORDER=PCI_BUS_ID is mandatory or index 4 lands on the T400.
#
# Usage: run_onsupport.sh <shard> <gpu>   shard = small | factual
set -u
SHARD=${1:?small|factual}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
case "$SHARD" in
  small)   C="--cap-neutral 200 --cap-creative 150 --cap-factual 0" ;;
  factual) C="--cap-neutral 0 --cap-creative 0 --cap-factual 500" ;;
  *) echo "unknown shard $SHARD" >&2; exit 2 ;;
esac
LOG=output/logs/onsup_$SHARD.log
rm -f output/onsup/$SHARD.done output/onsup/$SHARD.fail
mkdir -p output/onsup
echo "[onsup:$SHARD] START $(date +%H:%M:%S) gpu=$GPU tpp=256" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 256 $C \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --data-dir data --max-new-tokens 200 --batch-size 64 \
  --output-dir "output/onsup/sel_$SHARD" >> "$LOG" 2>&1
RC=$?
echo "[onsup:$SHARD] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch output/onsup/$SHARD.done || touch output/onsup/$SHARD.fail
