#!/usr/bin/env bash
# Status of every arm, read from the CURRENT run only.
#
# Every launcher appends to one log per cell, so a log can hold several runs and `grep DONE rc=` on
# the whole file reports the OLDEST failure as if it were the current state. That misread the state
# three times on 2026-09-22 -- once claiming feat-172 had died when it was 9,660 draws in, and once
# claiming five cotaeval cells had failed when they were running. The fix is to cut the log at its
# LAST START line and read only what follows.
#
# Usage: arm_status.sh [log-glob]     default: every log under output/logs
set -u
cd "$(dirname "$0")/.."
GLOB=${1:-'output/logs/*.log'}
printf "%-28s %-9s %s\n" ARM STATE DETAIL
for f in $GLOB; do
  b=$(basename "$f" .log)
  case "$b" in *queue*) continue ;; esac
  # Only the current run: everything after the last START marker (launchers all write one).
  cur=$(awk '/START/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$f" 2>/dev/null)
  [ -z "$cur" ] && cur=$(cat "$f" 2>/dev/null)
  done_line=$(printf %s "$cur" | grep -oE 'DONE rc=[0-9]+' | tail -1)
  prog=$(printf %s "$cur" | grep -oE 'processed [0-9]+/[0-9]+|[0-9]+/[0-9]+ *$' | tail -1)
  if [ -n "$done_line" ]; then
    rc=${done_line##*=}
    [ "$rc" = 0 ] && state=DONE || state=FAILED
  else
    state=RUNNING
  fi
  printf "%-28s %-9s %s\n" "$b" "$state" "${prog:-}"
done
