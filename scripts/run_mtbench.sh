#!/usr/bin/env bash
# results/onset_prediction_third_workload.md (feat-173): MT-Bench cells. Same protocol as
# feat-170's, on the 80-prompt factual slot (no "Complete the prefix:" header is prepended there,
# which is why the standard benchmarks go through that slot at all).
# Usage: run_mtbench.sh <cell> <gpu>   cell = draws | opponent | k10 | kcal:<k> | bind:<k>
set -u
CELL=${1:?cell}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
C="--cap-factual 80 --cap-neutral 0 --cap-creative 0 --cap-val 0 --cap-test 0 --cap-attack-train 0"
case "$CELL" in
  draws)     K=0.0;  T=256; OUT=output/mtb/sel_anchor256; TAG=draws ;;
  opponent)  K=-1.0; T=1;   OUT=output/mtb/baseline;      TAG=opponent ;;
  k10)       K=10.0; T=1;   OUT=output/mtb/conc_k10;      TAG=k10 ;;
  kcal:*)    K=${CELL#kcal:}; T=1; OUT=output/mtb/kcal_$(echo "$K"|tr -d '.'); TAG=kcal$(echo "$K"|tr -d '.') ;;
  bind:*)    K=${CELL#bind:}; T=1; OUT=output/mtb/conc_bind;                   TAG=bind ;;
  *) echo "unknown cell $CELL" >&2; exit 2 ;;
esac
LOG=output/logs/mtb_$TAG.log
rm -f ~/v/logs/mtb_$TAG.done ~/v/logs/mtb_$TAG.fail
echo "[mtb:$TAG] START $(date +%H:%M:%S) gpu=$GPU k=$K tpp=$T" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values "$K" --trajectories-per-prompt "$T" $C \
  --data-dir data/bench/mtbench --max-new-tokens 200 --batch-size 64 \
  --output-dir "$OUT" >> "$LOG" 2>&1
RC=$?
echo "[mtb:$TAG] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/mtb_$TAG.done || touch ~/v/logs/mtb_$TAG.fail
