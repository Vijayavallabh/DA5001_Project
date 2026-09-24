#!/usr/bin/env bash
# feat-181 on host B's eight idle H100s: one queue shell per card, its jobs in order (caution (x)),
# no PID captured anywhere. Each job is "<arm>:<start>:<count>" for scripts/run_n512_draws.sh.
# G0's three regenerations of trajectory 255 go first on cards 0, 4 and 5; they take minutes.
# The split balances feat-172's measured rates: Arm A 14.4 card-hours per 256 indices, small 4.4,
# factual about 10.
# Usage: run_n512_hostb.sh        (writes output/logs/n512_queue.log; "[n512] drained" when all end)
set -u
cd "$(dirname "$0")/.."
mkdir -p output/logs
q() { local gpu=$1; shift; for j in "$@"; do IFS=: read -r arm s c <<< "$j"; bash scripts/run_n512_draws.sh "$arm" "$s" "$c" "$gpu"; done; }
echo "[n512] start $(date '+%F %T')" >> output/logs/n512_queue.log
q 0 a:255:1 a:256:64 &
q 1 a:320:64 &
q 2 a:384:64 &
q 3 a:448:64 &
q 4 small:255:1 small:256:256 &
q 5 factual:255:1 factual:256:86 &
q 6 factual:342:85 &
q 7 factual:427:85 &
wait
echo "[n512] drained $(date '+%F %T')" >> output/logs/n512_queue.log
