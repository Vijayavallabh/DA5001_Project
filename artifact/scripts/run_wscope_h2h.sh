#!/usr/bin/env bash
# results/onset_prediction_workload_scope.md (feat-170) stage 3: the order-averaged head-to-head
# that produces B1 (Arm A) and B2 (Arm B). Anchor and selection come from the same n=64 draw pool,
# as every other pass in this project does.
# Usage: run_wscope_h2h.sh <a|b> <gpu>
set -u
ARM=${1:?a|b}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
case "$ARM" in
  a) SEL=output/wscope/a_sel; MET=output/wscope/a_conc; BASE=output/wscope/a_baseline; D=data;              K=10 ;;
  b) SEL=output/wscope/b_sel; MET=output/wscope/b_conc; BASE=output/wscope/b_baseline; D=data/bench/alpaca; K=1  ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
LOG=output/logs/wsh_$ARM.log
rm -f ~/v/logs/wsh_$ARM.done ~/v/logs/wsh_$ARM.fail
echo "[wsh:$ARM] START $(date +%H:%M:%S) gpu=$GPU k=$K" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_averaged_h2h.py \
  --judge microsoft/Phi-3.5-mini-instruct \
  --sel-dir "$SEL" --metered-dir "$MET" --anchor-dir "$SEL" --baseline-dir "$BASE" \
  --rewards "results/wscope_rewards64_$ARM.csv" --data-dir "$D" \
  --tag "_wscope_$ARM" --seed 7717 --n 64 --k "$K" --out results >> "$LOG" 2>&1
RC=$?
echo "[wsh:$ARM] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/wsh_$ARM.done || touch ~/v/logs/wsh_$ARM.fail
