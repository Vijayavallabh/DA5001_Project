#!/usr/bin/env bash
# results/onset_prediction_workload_scope.md (feat-170). Arm A is our own 850 ordinary prompts
# through feat-168's EXACT pipeline; Arm B re-draws AlpacaEval at a disjoint seed.
#
# --batch-size 64 is not a choice here, it is the thing being held fixed: the committed pass used
# 8 and feat-168 used 64, and batch size is part of the seed and a shift at a rate-valued quantity
# (cautions (u), (v)). Arm A exists precisely to remove that difference from the comparison.
#
# Usage: run_wscope.sh <cell> <gpu>
#   cell = a_neutral | a_creative | a_factual | a_metered | a_opponent
#        | b_draws | b_metered | b_opponent
set -u
CELL=${1:?cell}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/wscope_$CELL.log
rm -f ~/v/logs/wscope_$CELL.done ~/v/logs/wscope_$CELL.fail
# Caps are written out per cell rather than composed from a template. The first version built the
# Arm B lists with a ${Z#...} prefix trim that silently did nothing -- Z does not start with that
# prefix -- leaving a DUPLICATE --cap-factual where the later 0 wins, so Arm B would have generated
# nothing at all and looked like a fast success. Explicit beats clever in a launcher.

case "$CELL" in
  # Arm A, sharded by prompt class so three cards run in parallel without any PID being captured
  # (caution (x)). Each shard writes its own directory; the scorer reads all three.
  a_neutral)  D=data; K=0.0;  T=64; C="--cap-neutral 200 --cap-creative 0 --cap-factual 0 --cap-val 0 --cap-test 0 --cap-attack-train 0"; O=output/wscope/a_sel_neutral ;;
  a_creative) D=data; K=0.0;  T=64; C="--cap-neutral 0 --cap-creative 150 --cap-factual 0 --cap-val 0 --cap-test 0 --cap-attack-train 0"; O=output/wscope/a_sel_creative ;;
  a_factual)  D=data; K=0.0;  T=64; C="--cap-neutral 0 --cap-creative 0 --cap-factual 500 --cap-val 0 --cap-test 0 --cap-attack-train 0"; O=output/wscope/a_sel_factual ;;
  a_metered)  D=data; K=10.0; T=1;  C="--cap-neutral 200 --cap-creative 150 --cap-factual 500 --cap-val 0 --cap-test 0 --cap-attack-train 0"; O=output/wscope/a_conc ;;
  a_opponent) D=data; K=-1.0; T=1;  C="--cap-neutral 200 --cap-creative 150 --cap-factual 500 --cap-val 0 --cap-test 0 --cap-attack-train 0"; O=output/wscope/a_baseline ;;
  # Arm B: AlpacaEval again, disjoint seed. No bit-identity gate exists for this and none should
  # (feat-131): it is an independent draw by construction.
  b_draws)    D=data/bench/alpaca; K=0.0;  T=64; C="--cap-factual 805 --cap-neutral 0 --cap-creative 0 --cap-val 0 --cap-test 0 --cap-attack-train 0"; O=output/wscope/b_sel ;;
  b_metered)  D=data/bench/alpaca; K=1.0;  T=1;  C="--cap-factual 805 --cap-neutral 0 --cap-creative 0 --cap-val 0 --cap-test 0 --cap-attack-train 0"; O=output/wscope/b_conc ;;
  b_opponent) D=data/bench/alpaca; K=-1.0; T=1;  C="--cap-factual 805 --cap-neutral 0 --cap-creative 0 --cap-val 0 --cap-test 0 --cap-attack-train 0"; O=output/wscope/b_baseline ;;
  *) echo "unknown cell $CELL" >&2; exit 2 ;;
esac
SEEDS=""
case "$CELL" in b_*) SEEDS="--seeds 52" ;; esac

echo "[ws:$CELL] START $(date +%H:%M:%S) gpu=$GPU k=$K tpp=$T seeds='${SEEDS:-default}'" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values "$K" --trajectories-per-prompt "$T" $C $SEEDS \
  --data-dir "$D" --max-new-tokens 200 --batch-size 64 --output-dir "$O" >> "$LOG" 2>&1
RC=$?
echo "[ws:$CELL] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/wscope_$CELL.done || touch ~/v/logs/wscope_$CELL.fail
