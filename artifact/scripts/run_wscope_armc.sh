#!/usr/bin/env bash
# results/onset_prediction_workload_scope.md (feat-170) Arm C: the metered cell at the budget the
# committed argmin rule chose, then its head-to-head. One queue shell, sequenced by this shell
# (caution (x)); no PID is captured.
#
# k is passed in, not hardcoded, so the number in the run is the number the selector produced.
# The anchor draws and the opponent are Arm A's and are NOT re-run: they do not depend on k, and
# re-drawing them would change two things at once (caution (v)).
# Usage: run_wscope_armc.sh <k> <gpu>
set -u
K=${1:?k}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/wscope_c.log
rm -f ~/v/logs/wscope_c.done ~/v/logs/wscope_c.fail
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[wsc:C] METERED START $(date +%H:%M:%S) gpu=$GPU k=$K" >> "$LOG"
env $E .venv/bin/python h1.py --k-values "$K" --trajectories-per-prompt 1 \
  --cap-neutral 200 --cap-creative 150 --cap-factual 500 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --data-dir data --max-new-tokens 200 --batch-size 64 \
  --output-dir output/wscope/c_conc >> "$LOG" 2>&1
RC=$?
echo "[wsc:C] METERED rc=$RC $(date +%H:%M:%S)" >> "$LOG"
if [ $RC -ne 0 ]; then touch ~/v/logs/wscope_c.fail; exit $RC; fi

echo "[wsc:C] H2H START $(date +%H:%M:%S)" >> "$LOG"
env $E .venv/bin/python analysis/order_averaged_h2h.py \
  --judge microsoft/Phi-3.5-mini-instruct \
  --sel-dir output/wscope/a_sel --metered-dir output/wscope/c_conc \
  --anchor-dir output/wscope/a_sel --baseline-dir output/wscope/a_baseline \
  --rewards results/wscope_rewards64_a.csv --data-dir data \
  --tag "_wscope_c" --seed 7717 --n 64 --k "$K" --out results >> "$LOG" 2>&1
RC=$?
echo "[wsc:C] H2H rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/wscope_c.done || touch ~/v/logs/wscope_c.fail
