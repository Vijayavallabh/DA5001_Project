#!/usr/bin/env bash
# feat-166 stage 2: score the 51,520 AlpacaEval candidates once with the committed pointwise
# reward, so the judging stage reads a cache rather than re-scoring per judge.
set -u
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
rm -f ~/v/logs/mixpow_rewards.done ~/v/logs/mixpow_rewards.fail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-1} HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_scaling.py \
  --gen-dir output/mixpow/sel_anchor64 --baseline-dir output/mixpow/baseline \
  --reward-cache results/mixpow_rewards64.csv --tag "_mixpow" --max-n 64 \
  --judges microsoft/Phi-3.5-mini-instruct --batch-size 16 \
  --out results >> output/logs/mixpow_rewards.log 2>&1
RC=$?
echo "[mpr] DONE rc=$RC $(date +%H:%M:%S)" >> output/logs/mixpow_rewards.log
[ $RC -eq 0 ] && touch ~/v/logs/mixpow_rewards.done || touch ~/v/logs/mixpow_rewards.fail
