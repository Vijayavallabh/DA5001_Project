#!/usr/bin/env bash
# One queue shell per card, rungs in order (caution (x)); no PID is ever captured.
# Usage: run_oppalp_queue.sh <gpu> <model:tag> [<model:tag> ...]
set -u
GPU=${1:?gpu}; shift
cd "$(dirname "$0")/.."
for R in "$@"; do bash scripts/run_oppalp.sh "${R%%:*}" "${R##*:}" "$GPU"; done
echo "[oppalp queue gpu=$GPU] DRAINED $(date +%H:%M:%S)" >> output/logs/oppalp_queue$GPU.log
