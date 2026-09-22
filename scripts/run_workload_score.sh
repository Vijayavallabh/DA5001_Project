#!/usr/bin/env bash
# Score a workload arm: one reward pass, then the order-averaged head-to-head at each budget.
# Generic over corpus so a new workload cannot drift from feat-170's protocol.
#
# Usage: run_workload_score.sh <corpus> <datadir> <maxn> <gpu> <cell:k> [<cell:k> ...]
#   <datadir> is passed EXPLICITLY and not derived from <corpus>: the output tree and the corpus
#   directory do not always share a name (MT-Bench writes output/mtb/ from data/bench/mtbench/),
#   and deriving it silently pointed the judge at a directory that does not exist.
#   cell:k pairs the metered directory suffix with the k that names its trajectory files, e.g.
#   conc_k10:10 or conc_bind:1.4 -- h1.py writes the CLI k string verbatim into filenames
#   (caution (o)), so the k passed here must be the k the cell was generated with.
set -u
CORPUS=${1:?corpus}; DATA=${2:?datadir}; MAXN=${3:?maxn}; GPU=${4:?gpu}; shift 4
[ -d "$DATA" ] || { echo "no such corpus dir: $DATA" >&2; exit 2; }
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"
SEL=output/$CORPUS/sel_anchor256
[ -d "$SEL" ] || SEL=output/$CORPUS/sel_anchor64
LOG=output/logs/${CORPUS}_score.log
rm -f ~/v/logs/${CORPUS}_score.done ~/v/logs/${CORPUS}_score.fail

echo "[$CORPUS:score] REWARDS START $(date +%H:%M:%S) gpu=$GPU sel=$SEL maxn=$MAXN" >> "$LOG"
env $E .venv/bin/python analysis/selection_scaling.py \
  --gen-dir "$SEL" --baseline-dir "output/$CORPUS/baseline" \
  --reward-cache "results/${CORPUS}_rewards.csv" --tag "_$CORPUS" --max-n "$MAXN" \
  --judges microsoft/Phi-3.5-mini-instruct --batch-size 16 \
  --out results >> "$LOG" 2>&1
RC=$?
echo "[$CORPUS:score] REWARDS rc=$RC $(date +%H:%M:%S)" >> "$LOG"
if [ $RC -ne 0 ]; then touch ~/v/logs/${CORPUS}_score.fail; exit $RC; fi

for PAIR in "$@"; do
  CELL=${PAIR%%:*}; K=${PAIR##*:}
  echo "[$CORPUS:score] H2H $CELL k=$K START $(date +%H:%M:%S)" >> "$LOG"
  env $E .venv/bin/python analysis/order_averaged_h2h.py \
    --judge microsoft/Phi-3.5-mini-instruct \
    --sel-dir "$SEL" --metered-dir "output/$CORPUS/$CELL" \
    --anchor-dir "$SEL" --baseline-dir "output/$CORPUS/baseline" \
    --rewards "results/${CORPUS}_rewards.csv" --data-dir "$DATA" \
    --tag "_${CORPUS}_${CELL}" --seed 7717 --n 64 --k "$K" --out results >> "$LOG" 2>&1
  RC=$?
  echo "[$CORPUS:score] H2H $CELL rc=$RC $(date +%H:%M:%S)" >> "$LOG"
  [ $RC -ne 0 ] && { touch ~/v/logs/${CORPUS}_score.fail; exit $RC; }
done
touch ~/v/logs/${CORPUS}_score.done
