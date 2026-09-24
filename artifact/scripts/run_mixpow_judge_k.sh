#!/usr/bin/env bash
# results/onset_prediction_mixtral_power_k.md (feat-168) stage 3: judge the pass whose metered cell
# actually binds. Identical to run_mixpow_judge.sh except --metered-dir and --k, which is the whole
# difference between this arm and feat-166 -- the anchor draws, the opponent and the 51,520 reward
# scores are feat-166's, unchanged, because they do not depend on k (caution (v)).
# Usage: run_mixpow_judge_k.sh <judge> <tag> <cards> [--device-map auto]
set -u
JUDGE=${1:?judge}; TAG=${2:?tag}; CARDS=${3:?cards}; DM=${4:-}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
rm -f ~/v/logs/mpjk_$TAG.done ~/v/logs/mpjk_$TAG.fail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARDS HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_averaged_h2h.py --judge "$JUDGE" $DM \
  --sel-dir output/mixpow/sel_anchor64 --metered-dir output/mixpow/conc_k10 \
  --anchor-dir output/mixpow/sel_anchor64 --baseline-dir output/mixpow/baseline \
  --rewards results/mixpow_rewards64.csv --data-dir data/bench/alpaca \
  --tag "_mixpowk_$TAG" --seed 7717 --n 64 --k 1 --out results \
  > "output/logs/mpjk_$TAG.log" 2>&1
RC=$?
[ $RC -eq 0 ] && touch ~/v/logs/mpjk_$TAG.done || touch ~/v/logs/mpjk_$TAG.fail
