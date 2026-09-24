#!/usr/bin/env bash
# A generic workload cell, used by feat-174 and anything after it. Earlier arms each got their own
# near-identical launcher; this is the same protocol parameterised, so a new workload cannot drift
# from the one feat-170 established.
#
# Usage: run_workload.sh <corpus> <cap> <cell> <gpu>
#   corpus = a directory under data/bench/ ; cap = prompts in the factual slot
#   cell   = draws | opponent | k10 | kcal:<k> | bind:<k>
set -u
CORPUS=${1:?corpus}; CAP=${2:?cap}; CELL=${3:?cell}; GPU=${4:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
C="--cap-factual $CAP --cap-neutral 0 --cap-creative 0 --cap-val 0 --cap-test 0 --cap-attack-train 0"
case "$CELL" in
  draws)    K=0.0;  T=64; OUT=output/$CORPUS/sel_anchor64; TAG=draws ;;
  opponent) K=-1.0; T=1;  OUT=output/$CORPUS/baseline;     TAG=opponent ;;
  k10)      K=10.0; T=1;  OUT=output/$CORPUS/conc_k10;     TAG=k10 ;;
  kcal:*)   K=${CELL#kcal:}; T=1; OUT=output/$CORPUS/kcal_$(echo "$K"|tr -d '.'); TAG=kcal$(echo "$K"|tr -d '.') ;;
  bind:*)   K=${CELL#bind:}; T=1; OUT=output/$CORPUS/conc_bind;                   TAG=bind ;;
  *) echo "unknown cell $CELL" >&2; exit 2 ;;
esac
LOG=output/logs/${CORPUS}_$TAG.log
rm -f ~/v/logs/${CORPUS}_$TAG.done ~/v/logs/${CORPUS}_$TAG.fail
echo "[$CORPUS:$TAG] START $(date +%H:%M:%S) gpu=$GPU k=$K tpp=$T cap=$CAP" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values "$K" --trajectories-per-prompt "$T" $C \
  --data-dir "data/bench/$CORPUS" --max-new-tokens 200 --batch-size 64 \
  --output-dir "$OUT" >> "$LOG" 2>&1
RC=$?
echo "[$CORPUS:$TAG] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/${CORPUS}_$TAG.done || touch ~/v/logs/${CORPUS}_$TAG.fail
