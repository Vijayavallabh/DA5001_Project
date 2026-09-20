#!/usr/bin/env bash
# Keep the two hosts in sync and print what is running on both. Safe to run repeatedly.
#
# DIRECTIONS ARE DELIBERATE AND ASYMMETRIC:
#   code    local -> host B   (local is authoritative; edits are made here)
#   results host B -> local   (--update, so a newer local file is never clobbered)
# output/ is NOT synced: it is tens of GB, gitignored and host-specific. hf_cache/ never moves.
#
# Process detection matches on the EXECUTABLE FIELD and never on the absence of a pattern
# (caution (c)): `pgrep -f X` matches the shell running it, and a waiter keyed on `! pgrep` can
# never exit. Staleness is read off the filesystem, which is the evidence that works.
set -u
cd "$(dirname "$0")/.."
H=PrakashDGX_H2
R='~/v/DA5001_Project'
EX="--exclude .git --exclude __pycache__ --exclude hf_cache --exclude output --exclude .venv
    --exclude data/bench/cotaeval_raw --exclude '*.pyc' --exclude .pt_now.txt"

case "${1:-status}" in
  push)   # code out
    rsync -az $EX --include 'analysis/***' --include 'scripts/***' --include 'tests/***' \
          --include 'figures/***' --include '*.py' --include '*.sh' --include '*.md' \
          --include '*.json' --include '*/' --exclude '*' ./ "$H:$R/" && echo "[sync] code -> host B ok" ;;
  pull)   # results back, never clobbering a newer local file
    rsync -az --update "$H:$R/results/" results/ && echo "[sync] results <- host B ok" ;;
  both)   "$0" push && "$0" pull ;;
esac

echo
echo "=== LOCAL $(date +%H:%M:%S) ==="
ps -eo pid,etime,args --no-headers \
  | awk '$3 ~ /python$/ && (/h1\.py/ || /h2\.py/ || /analysis\//) {printf "  job  %-8s %-12s %s\n", $1, $2, substr($0, index($0,$3), 70)}'
ps -eo pid,args --no-headers \
  | awk '$2 ~ /bash$/ && /scripts\/run_/ {printf "  shell %-8s %s\n", $1, substr($0, index($0,$2), 60)}'
env -u LD_LIBRARY_PATH nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
  --format=csv,noheader 2>/dev/null | sed 's/^/  gpu   /' | head -5
for L in output/logs/comma7b128_card1.log output/logs/comma7b128_merge.log; do
  [ -f "$L" ] && printf "  log   %-42s %4d min since last write\n" \
    "$(basename "$L")" "$(( ($(date +%s) - $(stat -c %Y "$L")) / 60 ))"
done

echo
echo "=== HOST B ==="
ssh -o ConnectTimeout=15 "$H" 'nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | sed "s/^/  gpu   /"; echo "  --- sessions ---"; tmux -S ~/v/tmux.sock ls 2>/dev/null | sed "s/^/  tmux  /" || echo "  tmux  (none)"; echo "  --- markers ---"; ls -t ~/v/logs/*.done ~/v/logs/*.fail 2>/dev/null | head -4 | sed "s|.*/|  mark  |"'
