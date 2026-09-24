#!/usr/bin/env bash
# Wait for feat-134 to reach a terminal state, and report which one.
#
# Two defects this avoids. (1) A whole-file grep matches a HISTORICAL terminal state: card2b's first
# creative attempt failed at 12:38 and a second invocation started at 13:07, so `grep ABORT log`
# was true the moment it was armed. Only the text after the LAST "START creative" belongs to the
# run in flight. (2) Silence is not success: if the owner shell dies without writing its completion
# line the arm stalls forever and a success-only filter would never fire, so the owner PID is
# checked with `kill -0` (never pgrep -- caution (c)) and its disappearance is itself an event.
set -u
cd "$(dirname "$0")/.."
LOG=output/logs/comma7b128_card2.log
OWNER=${1:?usage: wait_comma7b128.sh <card2b pid>}

tail_of_run() { awk '/START creative/{b=""} {b=b $0 "\n"} END{printf "%s", b}' "$LOG"; }

while :; do
  if tail_of_run | grep -qE 'COMMA-7B n=128 ARM DRAINED'; then
    echo "feat-134 DRAINED -- generation, merge and scoring all finished"
    tail_of_run | grep -E 'merging|scoring rc=|DRAINED'
    exit 0
  fi
  if tail_of_run | grep -qE 'ABORT|generation rc=[1-9]'; then
    echo "feat-134 FAILED in the run that is in flight"
    tail_of_run | grep -E 'ABORT|generation rc=[1-9]'
    exit 0
  fi
  if ! kill -0 "$OWNER" 2>/dev/null; then
    echo "feat-134 OWNER SHELL $OWNER IS GONE and wrote no completion line -- the arm is stalled"
    echo "the three classes would need a new owner for the merge and the scoring"
    tail -3 "$LOG"
    exit 0
  fi
  sleep 300
done
