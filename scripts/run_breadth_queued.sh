#!/usr/bin/env bash
# Wait for a PID, then run one breadth anchor. Keeps one heavy h1 job per card: the first attempt
# put a 7B at batch 32 on a card already holding a 3B+8B pair, and the broken NVML reported the
# resulting OOM as an internal allocator assert rather than as an OOM.
set -u
GPU=$1; MODEL=$2; TAG=$3; WAIT_PID=${4:-}
LOG=output/logs/breadth${TAG}.log
if [ -n "$WAIT_PID" ]; then
  echo "[br${TAG}] waiting on PID $WAIT_PID at $(date +%H:%M)" >> "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  sleep 30
fi
exec ./scripts/run_breadth_anchor.sh "$GPU" "$MODEL" "$TAG"
