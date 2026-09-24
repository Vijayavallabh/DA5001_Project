#!/usr/bin/env bash
# Run a command once a named arm finishes, or give up.
#
# Usage: after.sh <marker-name> <deadline-minutes> -- <command> [args...]
#   waits for ~/v/logs/<marker-name>.done (success) or .fail, then runs the command only on .done.
#
# THE THREE WAYS THIS HAS GONE WRONG BEFORE, ALL FIXED HERE:
#   * caution (c): a waiter whose exit condition is `! pgrep -f X` can never exit, because the
#     polling shell's own command line contains X. This waits on the FILESYSTEM, never on the
#     absence of a process.
#   * caution (ay), the ninth incident: `ls A B` is an AND over two paths that are mutually
#     exclusive by design and exits 2 when either is missing, so a waiter obeying the filesystem
#     rule still polled for 19 hours. The wait below is an explicit OR (`test -e A -o -e B`) and
#     its exit status is checked against a mock directory in tests/test_after_waiter.py.
#   * the same caution: give every waiter a DEADLINE, so an impossible condition ends instead of
#     polling forever.
set -u
NAME=${1:?marker name}; DEADLINE_MIN=${2:?deadline minutes}; shift 2
[ "${1:-}" = "--" ] || { echo "usage: after.sh <name> <minutes> -- <command>" >&2; exit 2; }
shift
D="$HOME/v/logs/$NAME.done"; F="$HOME/v/logs/$NAME.fail"
END=$(( $(date +%s) + DEADLINE_MIN * 60 ))
while ! test -e "$D" -o -e "$F"; do
  if [ "$(date +%s)" -ge "$END" ]; then
    echo "[after:$NAME] DEADLINE $DEADLINE_MIN min reached with neither .done nor .fail; giving up" >&2
    exit 3
  fi
  sleep 60
done
if [ -e "$F" ]; then
  echo "[after:$NAME] the arm FAILED; not running the follow-up" >&2
  exit 4
fi
echo "[after:$NAME] .done at $(date +%H:%M:%S); running: $*"
exec "$@"
