#!/usr/bin/env bash
# Choose a workload's binding budget by the registered rule and run that cell, then score.
#
# The rule is argmin |activity(k) - <target>| over the calibration grid, and it is applied by
# analysis/budget_calibration.py rather than by a human reading a table -- which is the whole point
# of registering it. If the grid does not bracket the target the script STOPS: taking the nearest
# endpoint is what caution (g) forbids, and a refinement grid is a decision for the registration,
# not for a launcher.
#
# Usage: run_workload_bind.sh <corpus> <datadir> <cap> <maxn> <target> <gpu> [grid...]
#   [grid...] restricts the argmin to the named k values. A refinement is registered as a choice
#   over the REFINED grid alone, so an arm that has both grids on disk must say which one --
#   otherwise budget_calibration.py's default reselects the coarse points the refinement replaced.
set -uo pipefail
CORPUS=${1:?corpus}; DATA=${2:?datadir}; CAP=${3:?cap}; MAXN=${4:?maxn}; TARGET=${5:?target}; GPU=${6:?gpu}
shift 6; GRID=${*:+--grid $*}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
LOG=output/logs/${CORPUS}_bindpick.log
echo "[$CORPUS:bindpick] START $(date +%H:%M:%S) target=$TARGET grid=${GRID:-default}" >> "$LOG"

# CAUTION (ax): TWO PASSES MUST NOT SHARE A FILE. budget_calibration.py's default name follows
# --root, which is right for one grid per corpus and wrong the moment a refinement runs: the
# refined sweep wrote over the coarse one, deleting the CSV that is the EVIDENCE the refinement
# was needed. A named grid gets its own file.
KOUT="results/${CORPUS}_kcal.csv"
[ -n "$GRID" ] && KOUT="results/${CORPUS}_kcal_refined.csv"
OUT=$(.venv/bin/python analysis/budget_calibration.py --root "output/$CORPUS" \
        --target "$TARGET" $GRID --out "$KOUT" 2>&1) || { echo "$OUT" >> "$LOG"; exit 2; }
echo "$OUT" >> "$LOG"
echo "$OUT" | grep -q "G-cal PASS" || {
  echo "[$CORPUS:bindpick] G-cal FAILED; the binding cell is NOT RUN (caution (g))" >> "$LOG"
  exit 3
}
K=$(echo "$OUT" | sed -n 's/^CHOSEN k = \([0-9.]*\).*/\1/p')
[ -n "$K" ] || { echo "[$CORPUS:bindpick] could not read the chosen k" >> "$LOG"; exit 4; }
# BRACKETING IS NOT SUFFICIENT, and this script did not know it. feat-174 amended G-cal "for this
# arm and every later one": the chosen point must ALSO satisfy G0's 2x tolerance, and if no grid
# point does, the answer is REFINE rather than a choice. A rule that returns the nearest of five
# useless points is caution (p)'s gate-that-passes-everything wearing an argmin. Without this check
# the Gutenberg arm launched its binding cell at 0.47x the target on 2026-09-22.
RATIO=$(echo "$OUT" | sed -n 's/.*ratio \([0-9.]*\)x.*/\1/p')
[ -n "$RATIO" ] || { echo "[$CORPUS:bindpick] could not read the ratio" >> "$LOG"; exit 4; }
if ! .venv/bin/python -c "import sys; sys.exit(0 if 0.5 <= $RATIO <= 2.0 else 1)"; then
  echo "[$CORPUS:bindpick] REFINE: argmin k=$K is ${RATIO}x the target, outside G0's 2x band." >> "$LOG"
  echo "[$CORPUS:bindpick] The binding cell is NOT RUN; register a refined grid first." >> "$LOG"
  exit 6
fi
echo "[$CORPUS:bindpick] chosen k=$K at ${RATIO}x the target" >> "$LOG"

bash scripts/run_workload.sh "$CORPUS" "$CAP" "bind:$K" "$GPU" || exit 5
# h1.py writes the CLI k string verbatim into filenames (caution (o)), so the k handed to the
# scorer must be the same token this script passed to the generator.
bash scripts/run_workload_score.sh "$CORPUS" "$DATA" "$MAXN" "$GPU" "conc_k10:10" "conc_bind:$K"
