#!/usr/bin/env bash
# feat-166 stage 3: judge the AlpacaEval pass. Judge B on one card, Mixtral on two.
# Usage: run_mixpow_judge.sh <judge> <tag> <cards> [--device-map auto]
set -u
JUDGE=${1:?judge}; TAG=${2:?tag}; CARDS=${3:?cards}; DM=${4:-}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
rm -f ~/v/logs/mpj_$TAG.done ~/v/logs/mpj_$TAG.fail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARDS HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_averaged_h2h.py --judge "$JUDGE" $DM \
  --sel-dir output/mixpow/sel_anchor64 --metered-dir output/mixpow/conc_all \
  --anchor-dir output/mixpow/sel_anchor64 --baseline-dir output/mixpow/baseline \
  --rewards results/mixpow_rewards64.csv --data-dir data/bench/alpaca \
  --tag "_mixpow_$TAG" --seed 7717 --n 64 --k 10 --out results \
  > "output/logs/mpj_$TAG.log" 2>&1
RC=$?
[ $RC -eq 0 ] && touch ~/v/logs/mpj_$TAG.done || touch ~/v/logs/mpj_$TAG.fail
