#!/usr/bin/env bash
# Re-judge an EXISTING opponent arm under a different judge. Judging only: the opponent's
# completions and all four committed arms are on disk and nothing is regenerated, so the only
# thing that changes is the instrument.
#
# Usage: run_opponent_judge.sh <tag> <judge> <judge-tag> <cards> [--device-map auto]
set -uo pipefail
TAG=${1:?tag}; JUDGE=${2:?judge}; JT=${3:?judge tag}; CARDS=${4:?cards}; DM=${5:-}
# Derived, not hardcoded: a literal `$HOME/v/<project>` puts the repository name inside
# every launcher, and scripts/build_artifact.sh refuses to ship a tree containing it.
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
OUT=output/opponent_$TAG
[ -d "$OUT" ] || { echo "no such opponent run: $OUT" >&2; exit 2; }
MARK="$HOME/v/logs/oppj_${TAG}_${JT}"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CARDS"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
{
echo "[oppj:$TAG:$JT] judge=$JUDGE cards=$CARDS $(date +%H:%M)"
.venv/bin/python analysis/order_averaged_h2h.py --judge "$JUDGE" $DM \
  --sel-dir output/xfer/sel_anchor64 --metered-dir output/xfer/conc_all \
  --anchor-dir output/sweep_plain --baseline-dir "$OUT" \
  --rewards results/selection_rewards64.csv --n 64 --k 10 --seed 7717 \
  --tag "_opp_${TAG}_${JT}" --out results
echo "[oppj:$TAG:$JT] exit=$?"
} > "${MARK}.log" 2>&1
[ -f "results/order_averaged_h2h__opp_${TAG}_${JT}.csv" ] && touch "${MARK}.done" || touch "${MARK}.fail"
