#!/usr/bin/env bash
# feat-187: the k=0.5 judge pass OOM'd beside FActScore on host B GPU 7. Re-run it on the same card
# (same silicon as the other two passes, so G2 compares like with like) once FActScore has released
# it (OR wait over he_fact.done/.fail, with a deadline). Same command as run_anchoredbyte_judge.sh.
set -u
GPU=${1:?gpu}; K=${2:-0.5}
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
M=${AB_MARKS:-$HOME/v/logs}
dl=$(( $(date +%s) + 14400 ))
until [ -e "$M/he_fact.done" ] || [ -e "$M/he_fact.fail" ]; do [ "$(date +%s)" -ge "$dl" ] && exit 2; sleep 30; done
rm -f "$M/abj_k$K.done" "$M/abj_k$K.fail"
.venv/bin/python analysis/order_averaged_h2h.py --deecho --seed 7717 --n 64 --out results \
  --sel-dir output/phase5/sel_comma7b_64 --rewards results/selection_rewards64_comma7b.csv \
  --metered-dir output/anchoredbyte/comma7b_70b --k "$K" --anchor-dir output/phase5/sel_comma7b_64 \
  --baseline-dir output/sweep_plain --extra-dir output/phase5/imit_llama70b --extra-token=-1 \
  --extra-name risky70b --tag "ab70_k$K" > "output/logs/abj_k$K.log" 2>&1 \
  && touch "$M/abj_k$K.done" || touch "$M/abj_k$K.fail"
