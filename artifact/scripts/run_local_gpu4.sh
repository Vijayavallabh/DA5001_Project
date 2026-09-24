#!/usr/bin/env bash
# Local GPU 4, in order (caution (x)): feat-188's generation (seeds 82 83 84), its R0 (the seed-52 pool
# on record re-judged on recovered text), then feat-191's FActScore pass.
set -u
cd "$(dirname "$0")/.." || exit 1
M=output/logs/marks; mkdir -p $M
REPLIC_MARKS=$M REPLIC_GEN_ONLY=1 bash scripts/run_replic.sh a 4
. scripts/gpu_env.sh
rm -f $M/replic_R0.done $M/replic_R0.fail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_averaged_h2h.py --deecho --seed 7717 --n 64 --k 10 --out results \
  --sel-dir output/phase5/sel_anchor64_seed52 --rewards results/selection_rewards64_seed52.csv \
  --metered-dir output/phase2/conc_all --anchor-dir output/sweep_plain --baseline-dir output/sweep_plain \
  --tag seed52_deecho > output/logs/replic_R0.log 2>&1 && touch $M/replic_R0.done || touch $M/replic_R0.fail
HE_MARKS=$M bash scripts/run_he_metrics.sh factscore 4
