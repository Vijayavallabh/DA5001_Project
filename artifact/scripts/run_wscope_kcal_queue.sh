#!/usr/bin/env bash
# One queue shell per card, running its list in order (caution (x)): the sequencing is this
# shell's own and no PID is ever captured.
# Usage: run_wscope_kcal_queue.sh <gpu> <k> [<k> ...]
set -u
GPU=${1:?gpu}; shift
cd "$(dirname "$0")/.."
for K in "$@"; do bash scripts/run_wscope_kcal.sh "$K" "$GPU"; done
echo "[queue gpu=$GPU] DRAINED $(date +%H:%M:%S)" >> output/logs/wskcal_queue$GPU.log
