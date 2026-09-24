#!/usr/bin/env bash
# results/onset_prediction_mixtral_armc.md (feat-171): re-judge feat-170 Arm C. No generation, no
# re-scoring -- the directories and the 54,400-candidate reward cache are Arm C's, and only the
# judge changes (caution (v): never move two things at once).
# Usage: run_armc_judge.sh <judge> <tag> <cards> [--device-map auto]
set -u
JUDGE=${1:?judge}; TAG=${2:?tag}; CARDS=${3:?cards}; DM=${4:-}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
rm -f ~/v/logs/acj_$TAG.done ~/v/logs/acj_$TAG.fail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARDS HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_averaged_h2h.py --judge "$JUDGE" $DM \
  --sel-dir output/wscope/a_sel --metered-dir output/wscope/c_conc \
  --anchor-dir output/wscope/a_sel --baseline-dir output/wscope/a_baseline \
  --rewards results/wscope_rewards64_a.csv --data-dir data \
  --tag "_armc_$TAG" --seed 7717 --n 64 --k 0.9 --out results \
  > "output/logs/acj_$TAG.log" 2>&1
RC=$?
echo "[acj:$TAG] DONE rc=$RC $(date +%H:%M:%S)" >> "output/logs/acj_$TAG.log"
[ $RC -eq 0 ] && touch ~/v/logs/acj_$TAG.done || touch ~/v/logs/acj_$TAG.fail
