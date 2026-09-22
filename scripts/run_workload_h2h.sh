#!/usr/bin/env bash
# One order-averaged head-to-head for a workload arm under a named judge. Judging only: the
# generations and the reward cache already exist, and only the judge changes (caution (v)).
# Usage: run_workload_h2h.sh <corpus> <datadir> <cell> <k> <judge> <tag> <cards> [--device-map auto]
set -u
CORPUS=${1:?corpus}; DATA=${2:?datadir}; CELL=${3:?cell}; K=${4:?k}
JUDGE=${5:?judge}; TAG=${6:?tag}; CARDS=${7:?cards}; DM=${8:-}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
SEL=output/$CORPUS/sel_anchor256
[ -d "$SEL" ] || SEL=output/$CORPUS/sel_anchor64
[ -d "$DATA" ] || { echo "no such corpus dir: $DATA" >&2; exit 2; }
LOG=output/logs/${CORPUS}_h2h_${CELL}_${TAG}.log
rm -f ~/v/logs/${CORPUS}_h2h_${CELL}_${TAG}.done ~/v/logs/${CORPUS}_h2h_${CELL}_${TAG}.fail
echo "[$CORPUS:$CELL:$TAG] START $(date +%H:%M:%S) cards=$CARDS k=$K" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARDS HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_averaged_h2h.py --judge "$JUDGE" $DM \
  --sel-dir "$SEL" --metered-dir "output/$CORPUS/$CELL" \
  --anchor-dir "$SEL" --baseline-dir "output/$CORPUS/baseline" \
  --rewards "results/${CORPUS}_rewards.csv" --data-dir "$DATA" \
  --tag "_${CORPUS}_${CELL}_${TAG}" --seed 7717 --n 64 --k "$K" --out results >> "$LOG" 2>&1
RC=$?
echo "[$CORPUS:$CELL:$TAG] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/${CORPUS}_h2h_${CELL}_${TAG}.done || touch ~/v/logs/${CORPUS}_h2h_${CELL}_${TAG}.fail
