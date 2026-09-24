#!/usr/bin/env bash
# One queue shell per card, cells in order (caution (x)); no PID is ever captured.
# Usage: run_workload_queue.sh <corpus> <cap> <gpu> <cell> [<cell> ...]
set -u
CORPUS=${1:?corpus}; CAP=${2:?cap}; GPU=${3:?gpu}; shift 3
cd "$(dirname "$0")/.."
for C in "$@"; do bash scripts/run_workload.sh "$CORPUS" "$CAP" "$C" "$GPU"; done
echo "[$CORPUS queue gpu=$GPU] DRAINED $(date +%H:%M:%S)" >> output/logs/${CORPUS}_queue$GPU.log
