#!/usr/bin/env bash
# feat-167 Half B judging: one auxiliary rung, judged in the same construction as feat-165's.
# Usage: run_ts_rung_judge.sh <tag> <gpu>
set -u
TAG=${1:?tag}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
rm -f ~/v/logs/tsrj_$TAG.done ~/v/logs/tsrj_$TAG.fail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_averaged_h2h.py --extra-dir "output/tokenswap/util_$TAG" \
  --extra-token tokenswap --extra-name "ts_$TAG" --tag "_ts_$TAG" --seed 7717 --n 64 --k 10 \
  --out results > "output/logs/tsrj_$TAG.log" 2>&1
RC=$?
[ $RC -eq 0 ] && touch ~/v/logs/tsrj_$TAG.done || touch ~/v/logs/tsrj_$TAG.fail
