#!/usr/bin/env bash
# Host B, 2026-09-24: hand GPUs 5-7 from the de-echo re-judge straight to the next jobs, so no card
# sits idle between them. Waits are on marker FILES with a deadline, never on a process pattern
# (caution (c)).  Usage: setsid nohup bash scripts/run_hostb_chain.sh > output/logs/hostb_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.." || exit 1
L=~/v/logs
waitm() { local dl=$(( $(date +%s) + ${2:-21600} )); until [ -e "$L/$1" ]; do
  [ "$(date +%s)" -ge "$dl" ] && { echo "[chain] gave up on $1 $(date '+%T')"; return 1; }; sleep 30; done; }
waitm dr_queue_q56.finished && { echo "[chain] q56 done $(date '+%T'); ladder re-runs on GPU 5"
  DR_GPU=5 bash scripts/run_deecho_rejudge.sh q5b; }
waitm dr_queue_q7.finished && echo "[chain] q7 done $(date '+%T')"
echo "[chain] AnchoredByte on 5,6,7 $(date '+%T')"
bash scripts/run_anchoredbyte.sh 5,6,7 "0.5 0.1 2"
echo "[chain] finished $(date '+%T')"
