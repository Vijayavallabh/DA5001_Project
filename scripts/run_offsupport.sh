#!/usr/bin/env bash
# results/onset_prediction_offsupport_ladder.md (feat-172): the anchor at n=256 on AlpacaEval.
#
# --batch-size 64 and every other flag match feat-166's sel_anchor64 exactly. Batch size is part
# of the seed (caution (u)) and matching it is what makes the reproduction gate possible at all:
# ranks 0-63 of this pool must be bit-identical to results/mixpow_rewards64.csv. Do not change it.
# One run yields n = 64, 128 and 256 by the prefix property feat-134 demonstrated.
# Usage: run_offsupport.sh <gpu>
set -u
GPU=${1:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/offsup_draws.log
rm -f ~/v/logs/offsup_draws.done ~/v/logs/offsup_draws.fail
echo "[offsup] START $(date +%H:%M:%S) gpu=$GPU tpp=256" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 256 \
  --cap-factual 805 --cap-neutral 0 --cap-creative 0 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --data-dir data/bench/alpaca --max-new-tokens 200 --batch-size 64 \
  --output-dir output/offsup/sel_anchor256 >> "$LOG" 2>&1
RC=$?
echo "[offsup] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/offsup_draws.done || touch ~/v/logs/offsup_draws.fail
