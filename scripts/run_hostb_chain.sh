#!/usr/bin/env bash
# Host B, 2026-09-24: hand GPUs 5-7 from the de-echo re-judge (and GPU 6 from the Prometheus pass)
# straight to AnchoredByte, so no card sits idle between them. Waits are on marker FILES with a
# deadline, never on a process pattern (caution (c)); a two-state job is waited on as .done OR .fail.
# Usage: setsid nohup bash scripts/run_hostb_chain.sh > output/logs/hostb_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.." || exit 1
L=~/v/logs
waitm() {  # waitm <marker> [<alt marker>]
  local dl=$(( $(date +%s) + 21600 ))
  until [ -e "$L/$1" ] || { [ -n "${2:-}" ] && [ -e "$L/$2" ]; }; do
    [ "$(date +%s)" -ge "$dl" ] && { echo "[chain] gave up on $1 $(date '+%T')"; return 1; }; sleep 30; done
  echo "[chain] $1 ${2:+or $2 }seen $(date '+%T')"; }
waitm dr_queue_q7.finished
waitm he_prom_a.done he_prom_a.fail
waitm he_prom_b.done he_prom_b.fail
echo "[chain] AnchoredByte on 5,6,7 $(date '+%T')"
bash scripts/run_anchoredbyte.sh 5,6,7 "0.5 0.1 2"
echo "[chain] finished $(date '+%T')"
