#!/usr/bin/env bash
# Detached status+sync heartbeat for both hosts. Appends to output/logs/monitor.log.
# Pulls results from host B each tick with --update, so a newer local file is never clobbered.
# Exits on its own after MAX ticks so it can never become a forgotten daemon.
set -u
cd "$(dirname "$0")/.."
EVERY=${1:-600}
MAX=${2:-144}
L=output/logs/monitor.log
mkdir -p output/logs
for i in $(seq 1 "$MAX"); do
  { echo "===== tick $i  $(date '+%Y-%m-%d %H:%M:%S') ====="
    bash scripts/sync_status.sh pull 2>&1 | grep -E '^\[sync\]' || echo "[sync] pull failed"
    bash scripts/sync_status.sh status 2>&1
  } >> "$L" 2>&1
  sleep "$EVERY"
done
echo "[monitor] finished $MAX ticks $(date)" >> "$L"
