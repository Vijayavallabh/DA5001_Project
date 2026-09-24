#!/usr/bin/env bash
# feat-188: R1b, R2, R2b in parallel on three host-B cards (same silicon as R1), instead of in series
# after R1 as run_replic_post.sh did. Same commands. Usage: run_replic_post_parallel.sh 3 5 6
set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
M=${REPLIC_MARKS:-$HOME/v/logs}; R=output/replic
H2H=".venv/bin/python analysis/order_averaged_h2h.py --deecho --seed 7717 --n 64 --k 10 --out results \
  --sel-dir $R/sel_anchor64 --metered-dir $R/conc_k10 --anchor-dir $R/anchor_k0"
job() { local n=$1 g=$2; shift 2; rm -f $M/replic_$n.done $M/replic_$n.fail
  if CUDA_VISIBLE_DEVICES=$g "$@" >> output/logs/replic_$n.log 2>&1; then touch $M/replic_$n.done; else touch $M/replic_$n.fail; fi; }
.venv/bin/python analysis/nonempty_rewards.py results/selection_rewards64_replic.csv \
  results/selection_rewards64_replic_nonempty.csv > output/logs/replic_nonempty.log 2>&1 || exit 1
job judge_R1b "$1" $H2H --rewards results/selection_rewards64_replic_nonempty.csv --baseline-dir output/sweep_plain --tag replic_nonempty &
job judge_R2 "$2" $H2H --rewards results/selection_rewards64_replic.csv --baseline-dir $R/opp --tag replic_opp &
job judge_R2b "$3" $H2H --rewards results/selection_rewards64_replic_nonempty.csv --baseline-dir $R/opp --tag replic_opp_nonempty &
wait
