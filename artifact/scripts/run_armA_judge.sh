#!/usr/bin/env bash
# feat-170 Arm A, second judge. Arm A (our corpus at the paper's own k=10) is the only cell of the
# judge x workload x budget grid with one judge on it; Arms B and C and the AlpacaEval binding pass
# all have two. Judging only -- the generations and the reward cache are Arm A's.
# Usage: run_armA_judge.sh <judge> <tag> <cards>
set -u
JUDGE=${1:?judge}; TAG=${2:?tag}; CARDS=${3:?cards}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
rm -f ~/v/logs/aaj_$TAG.done ~/v/logs/aaj_$TAG.fail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARDS HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_averaged_h2h.py --judge "$JUDGE" \
  --sel-dir output/wscope/a_sel --metered-dir output/wscope/a_conc \
  --anchor-dir output/wscope/a_sel --baseline-dir output/wscope/a_baseline \
  --rewards results/wscope_rewards64_a.csv --data-dir data \
  --tag "_armA_$TAG" --seed 7717 --n 64 --k 10 --out results \
  > "output/logs/aaj_$TAG.log" 2>&1
RC=$?
echo "[aaj:$TAG] DONE rc=$RC $(date +%H:%M:%S)" >> "output/logs/aaj_$TAG.log"
[ $RC -eq 0 ] && touch ~/v/logs/aaj_$TAG.done || touch ~/v/logs/aaj_$TAG.fail
