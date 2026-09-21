#!/usr/bin/env bash
# results/onset_prediction_mixtral_power.md (feat-166): a fresh, larger pass on AlpacaEval-805 so
# that Mixtral's UNRESOLVED reading can be told apart from a wide interval.
# One cell per card, no PID capture, the sequencing is this shell's (caution (x)).
# Usage: run_mixtral_power.sh <cell>   where cell is draws | metered | opponent
set -u
CELL=${1:?draws|metered|opponent}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
D=data/bench/alpaca
CAPS="--cap-factual 805 --cap-neutral 0 --cap-creative 0 --cap-val 0 --cap-test 0 --cap-attack-train 0"
LOG=output/logs/mixpow_$CELL.log
# Clear OUR OWN markers here rather than trusting the caller to: feat-165's first launch
# failed and left a .fail that survived the successful relaunch, so both sentinels sat
# beside each other and any waiter reading either one was right by luck (caution (c)).
rm -f ~/v/logs/mixpow_$CELL.done ~/v/logs/mixpow_$CELL.fail

case "$CELL" in
  draws)    GPU=3; K=0.0;   TPP=64; OUT=output/mixpow/sel_anchor64 ;;
  metered)  GPU=4; K=10.0;  TPP=1;  OUT=output/mixpow/conc_all ;;
  opponent) GPU=5; K=-1.0;  TPP=1;  OUT=output/mixpow/baseline ;;
  *) echo "unknown cell $CELL" >&2; exit 2 ;;
esac

echo "[mp:$CELL] START $(date +%H:%M:%S) gpu=$GPU k=$K tpp=$TPP" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values "$K" --trajectories-per-prompt "$TPP" $CAPS \
  --data-dir "$D" --max-new-tokens 200 --batch-size 64 --output-dir "$OUT" >> "$LOG" 2>&1
RC=$?
echo "[mp:$CELL] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/mixpow_$CELL.done || touch ~/v/logs/mixpow_$CELL.fail
