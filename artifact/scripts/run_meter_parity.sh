#!/usr/bin/env bash
# feat-159: give the METERED decoder the same reward model.
# Bands and gates: results/onset_prediction_meter_parity.md, committed before any generation.
#
# The protocol line is COPIED CHARACTER FOR CHARACTER from scripts/run_tqa_headtohead.sh, which
# produced results/verifiable_metered_tqa.csv, and exactly two things differ: --k-values is one
# budget instead of six, and --trajectories-per-prompt is 16 instead of 1. Batch size is held at
# 48 because batch size is part of the seed (cautions (u) and (v)) and must not move too.
#
# G3 in the registration is distributional and NOT bit-identity, precisely because 16 draws
# consume the RNG differently from 1 -- rank 0 here is not the committed trajectory, and a gate
# demanding that it were would be incoherent rather than strict (the feat-131 precedent).
#
# One queue shell per card running its jobs in order; no PID is ever captured (caution (x)).
# Usage: scripts/run_meter_parity.sh <gpu> <k> [extra-dir-to-reward-score]
set -u
GPU="${1:?usage: $0 <gpu> <k> [extra-dir]}"
K="${2:?usage: $0 <gpu> <k> [extra-dir]}"
EXTRA="${3:--}"          # explicit sentinel, never "" (caution (x))
cd "$(dirname "$0")/.."
mkdir -p output/logs "$HOME/v/logs"
TAG="parity_k${K}"
LOG="output/logs/${TAG}.log"
DIR="output/phase5/tqaP_k${K}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
COMMON="--data-dir data/bench/triviaqa --safe-model-path jacquelinehe/tinycomma-1.8b-llama3-tokenizer --risky-model-path meta-llama/Llama-3.1-8B-Instruct --cap-factual 500 --cap-neutral 0 --cap-creative 0 --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 24 --batch-size 48"

echo "[$TAG] $(date +%H:%M:%S) START k=$K gpu=$GPU" >> "$LOG"
if [ ! -f "$DIR/GEN_DONE" ]; then
  $PY h1.py --k-values "$K" --trajectories-per-prompt 16 $COMMON --output-dir "$DIR" >> "$LOG" 2>&1
  RC=$?
  echo "[$TAG] $(date +%H:%M:%S) generate exit=$RC" >> "$LOG"
  [ $RC -ne 0 ] && { touch "$HOME/v/logs/${TAG}.fail"; exit $RC; }
  touch "$DIR/GEN_DONE"
fi

$PY analysis/meter_parity.py --score-dir "$DIR" --tag "k${K}" --out results >> "$LOG" 2>&1
RC=$?
echo "[$TAG] $(date +%H:%M:%S) reward exit=$RC" >> "$LOG"
[ $RC -ne 0 ] && { touch "$HOME/v/logs/${TAG}.fail"; exit $RC; }

# The selection side, same scorer and same call, on generations that already exist.
if [ "$EXTRA" != "-" ]; then
  echo "[$TAG] $(date +%H:%M:%S) reward-scoring $EXTRA" >> "$LOG"
  $PY analysis/meter_parity.py --score-dir "$EXTRA" --tag "sel" --out results >> "$LOG" 2>&1
  echo "[$TAG] $(date +%H:%M:%S) extra exit=$?" >> "$LOG"
fi
touch "$HOME/v/logs/${TAG}.done"
