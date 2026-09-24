#!/usr/bin/env bash
# feat-187 (results/onset_prediction_anchoredbyte.md): the authors' AnchoredByte, Comma-7B + the 70B
# base, on host B cards <a>,<b>,<c> (safe model + the 70B's embedding/head on the first, its decoder
# layers on the other two). One budget after another; batch 32, retried at 16 only on OOM.
# Usage (host B): setsid nohup bash scripts/run_anchoredbyte.sh 0,1,2 "0.5 0.1 2" > /dev/null 2>&1 < /dev/null &
set -u
GPUS=${1:?gpus}; KS=${2:-0.5 0.1 2}
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
mkdir -p output/logs ~/v/logs
OUT=output/anchoredbyte/comma7b_70b
for K in $KS; do
  LOG=output/logs/ab70_k$K.log
  rm -f ~/v/logs/ab70_k$K.done ~/v/logs/ab70_k$K.fail
  for BS in 32 16; do
    echo "[ab70] k=$K batch=$BS start $(date '+%F %T')" >> "$LOG"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/anchoredbyte_decode.py --risky unsloth/Meta-Llama-3.1-70B \
      --k "$K" --batch-size "$BS" --out-dir "$OUT" >> "$LOG" 2>&1
    RC=$?
    echo "[ab70] k=$K batch=$BS exit=$RC $(date '+%F %T')" >> "$LOG"
    [ $RC -eq 0 ] && break
    grep -q 'OutOfMemoryError\|CUDA out of memory' "$LOG" || break
  done
  [ $RC -eq 0 ] && touch ~/v/logs/ab70_k$K.done || touch ~/v/logs/ab70_k$K.fail
done
