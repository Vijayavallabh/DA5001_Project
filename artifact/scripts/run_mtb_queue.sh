#!/usr/bin/env bash
# One queue shell per card, list in order (caution (x)). No PID is ever captured.
set -u
GPU=${1:?gpu}; shift
cd "$(dirname "$0")/.."
for C in "$@"; do bash scripts/run_mtbench.sh "$C" "$GPU"; done
echo "[mtb queue gpu=$GPU] DRAINED $(date +%H:%M:%S)" >> output/logs/mtb_queue$GPU.log
