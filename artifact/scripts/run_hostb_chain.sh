#!/usr/bin/env bash
# Host B, 2026-09-24 (the other project's job released GPUs 0-3 at ~14:35 IST): AnchoredByte k=0.1
# on 3,5,6 once the Prometheus halves free 5 and 6, then the 70B timing cells on the same three
# cards once the single-card timing cells (GPU 7) are done. Marker-file waits with deadlines,
# ORs over .done/.fail (caution (c)).
# Usage: setsid nohup bash scripts/run_hostb_chain.sh > output/logs/hostb_chain.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.." || exit 1
L=~/v/logs
waitm() {  # waitm <marker> [<alt marker>]
  local dl=$(( $(date +%s) + 21600 ))
  until [ -e "$L/$1" ] || { [ -n "${2:-}" ] && [ -e "$L/$2" ]; }; do
    [ "$(date +%s)" -ge "$dl" ] && { echo "[chain] gave up on $1 $(date '+%T')"; return 1; }; sleep 30; done
  echo "[chain] $1 ${2:+or $2 }seen $(date '+%T')"; }
waitm he_prom_a.done he_prom_a.fail
waitm he_prom_b.done he_prom_b.fail
echo "[chain] AnchoredByte k=0.1 on 3,5,6 $(date '+%T')"
bash scripts/run_anchoredbyte.sh 3,5,6 "0.1"
waitm blat_single.done blat_single.fail
echo "[chain] 70B timing on 3,5,6 $(date '+%T')"
bash scripts/run_batched_latency.sh 3 3,5,6 - 70b
echo "[chain] finished $(date '+%T')"
