#!/usr/bin/env bash
# results/onset_prediction_workload_scope.md (feat-170) stage 2: reward cache + judge B, per arm.
# Arm A's selection draws are sharded by prompt class across three directories; output/wscope/a_sel
# links the three NON-EMPTY class files into one tree. Linking every file instead clobbers the real
# neutral and creative shards with the empty ones each other shard also writes.
# Usage: run_wscope_score.sh <a|b> <gpu>
set -u
ARM=${1:?a|b}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
case "$ARM" in
  a) SEL=output/wscope/a_sel; BASE=output/wscope/a_baseline; D=data ;;
  b) SEL=output/wscope/b_sel; BASE=output/wscope/b_baseline; D=data/bench/alpaca ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
LOG=output/logs/wsc_$ARM.log
rm -f ~/v/logs/wsc_$ARM.done ~/v/logs/wsc_$ARM.fail
echo "[wsc:$ARM] START $(date +%H:%M:%S) gpu=$GPU" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_scaling.py \
  --gen-dir "$SEL" --baseline-dir "$BASE" \
  --reward-cache "results/wscope_rewards64_$ARM.csv" --tag "_wscope_$ARM" --max-n 64 \
  --judges microsoft/Phi-3.5-mini-instruct --batch-size 16 \
  --out results >> "$LOG" 2>&1
RC=$?
echo "[wsc:$ARM] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/wsc_$ARM.done || touch ~/v/logs/wsc_$ARM.fail
