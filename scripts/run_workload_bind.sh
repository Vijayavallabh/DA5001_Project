#!/usr/bin/env bash
# Choose a workload's binding budget by the registered rule and run that cell, then score.
#
# The rule is argmin |activity(k) - <target>| over the calibration grid, and it is applied by
# analysis/budget_calibration.py rather than by a human reading a table -- which is the whole point
# of registering it. If the grid does not bracket the target the script STOPS: taking the nearest
# endpoint is what caution (g) forbids, and a refinement grid is a decision for the registration,
# not for a launcher.
#
# Usage: run_workload_bind.sh <corpus> <datadir> <cap> <maxn> <target> <gpu>
set -uo pipefail
CORPUS=${1:?corpus}; DATA=${2:?datadir}; CAP=${3:?cap}; MAXN=${4:?maxn}; TARGET=${5:?target}; GPU=${6:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
LOG=output/logs/${CORPUS}_bindpick.log
echo "[$CORPUS:bindpick] START $(date +%H:%M:%S) target=$TARGET" >> "$LOG"

OUT=$(.venv/bin/python analysis/budget_calibration.py --root "output/$CORPUS" \
        --target "$TARGET" 2>&1) || { echo "$OUT" >> "$LOG"; exit 2; }
echo "$OUT" >> "$LOG"
echo "$OUT" | grep -q "G-cal PASS" || {
  echo "[$CORPUS:bindpick] G-cal FAILED; the binding cell is NOT RUN (caution (g))" >> "$LOG"
  exit 3
}
K=$(echo "$OUT" | sed -n 's/^CHOSEN k = \([0-9.]*\).*/\1/p')
[ -n "$K" ] || { echo "[$CORPUS:bindpick] could not read the chosen k" >> "$LOG"; exit 4; }
echo "[$CORPUS:bindpick] chosen k=$K" >> "$LOG"

bash scripts/run_workload.sh "$CORPUS" "$CAP" "bind:$K" "$GPU" || exit 5
# h1.py writes the CLI k string verbatim into filenames (caution (o)), so the k handed to the
# scorer must be the same token this script passed to the generator.
bash scripts/run_workload_score.sh "$CORPUS" "$DATA" "$MAXN" "$GPU" "conc_k10:10" "conc_bind:$K"
