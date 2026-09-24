#!/usr/bin/env bash
# feat-188's remaining judge passes, host B, after queue b writes R1's marker (OR, with a deadline):
# R1b (committed opponent, empty candidates excluded), R2 (re-drawn opponent), R2b (both).
# Usage: run_replic_post.sh <gpu>
set -u
GPU=${1:?gpu}
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
M=${REPLIC_MARKS:-$HOME/v/logs}; R=output/replic
dl=$(( $(date +%s) + 43200 ))
until [ -e $M/replic_judge_R1.done ] || [ -e $M/replic_judge_R1.fail ]; do [ "$(date +%s)" -ge "$dl" ] && exit 2; sleep 60; done
H2H=".venv/bin/python analysis/order_averaged_h2h.py --deecho --seed 7717 --n 64 --k 10 --out results \
  --sel-dir $R/sel_anchor64 --metered-dir $R/conc_k10 --anchor-dir $R/anchor_k0"
job() { local n=$1; shift; rm -f $M/replic_$n.done $M/replic_$n.fail
  if "$@" >> output/logs/replic_$n.log 2>&1; then touch $M/replic_$n.done; else touch $M/replic_$n.fail; fi; }
.venv/bin/python analysis/nonempty_rewards.py results/selection_rewards64_replic.csv \
  results/selection_rewards64_replic_nonempty.csv > output/logs/replic_nonempty.log 2>&1
job judge_R1b $H2H --rewards results/selection_rewards64_replic_nonempty.csv --baseline-dir output/sweep_plain --tag replic_nonempty
job judge_R2  $H2H --rewards results/selection_rewards64_replic.csv --baseline-dir $R/opp --tag replic_opp
job judge_R2b $H2H --rewards results/selection_rewards64_replic_nonempty.csv --baseline-dir $R/opp --tag replic_opp_nonempty
