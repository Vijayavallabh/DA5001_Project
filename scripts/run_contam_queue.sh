#!/usr/bin/env bash
# feat-112: run several contaminated-anchor arms in sequence on ONE card, in one shell.
# Chaining separate waiter shells and capturing their PIDs with ps|grep|awk mis-assigned the
# arguments on 2026-09-14 -- an empty wait-pid shifted the batch size into its slot and three arms
# launched with no wait at all, oversubscribing two cards. One shell per card, a list of jobs, no
# PID capture: the sequencing is the shell's own.
# Usage: run_contam_queue.sh <gpu> <wait-pid-or-"-"> <dir:tag:batch> [<dir:tag:batch> ...]
set -u
GPU=$1; WAIT_PID=$2; shift 2
if [ "$WAIT_PID" != "-" ]; then
  echo "[queue gpu$GPU] waiting on PID $WAIT_PID at $(date +%H:%M)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  sleep 30
fi
for job in "$@"; do
  DIR=${job%%:*}; rest=${job#*:}; TAG=${rest%%:*}; BATCH=${rest##*:}
  echo "[queue gpu$GPU] -> $TAG (batch $BATCH) at $(date +%H:%M)"
  ./scripts/run_contaminated_anchor.sh "$GPU" "$DIR" "$TAG" "" "$BATCH"
done
echo "[queue gpu$GPU] done at $(date +%H:%M)"
