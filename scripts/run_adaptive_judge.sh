#!/usr/bin/env bash
# results/adaptive_n_note.md: the registered judged pass for the adaptive rules, on one card after the
# card's previous queue writes its last marker (OR over .done/.fail, with a deadline).
# Usage: run_adaptive_judge.sh <gpu> [<marker>]
set -u
GPU=${1:?gpu}; AFTER=${2:-}
cd "$(dirname "$0")/.." || exit 1
M=output/logs/marks; mkdir -p $M
if [ -n "$AFTER" ]; then dl=$(( $(date +%s) + 28800 ))
  until [ -e "$M/$AFTER.done" ] || [ -e "$M/$AFTER.fail" ]; do [ "$(date +%s)" -ge "$dl" ] && exit 2; sleep 60; done; fi
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
P=output/phase5/sel_anchor64; R=results
rm -f $M/adaptive_judge.done $M/adaptive_judge.fail
.venv/bin/python analysis/levels_pass.py --tag adaptive \
  --arm sel64=sel:$P:$R/selection_rewards64.csv:64 \
  --arm ada64_q75=pick:$P:$R/adaptive_picks_ada64_q75.csv --arm fix7=pick:$P:$R/adaptive_picks_fix7_for_q75.csv \
  --arm ada64_q90=pick:$P:$R/adaptive_picks_ada64_q90.csv --arm fix17=pick:$P:$R/adaptive_picks_fix17_for_q90.csv \
  > output/logs/adaptive_judge.log 2>&1 \
&& .venv/bin/python analysis/levels_pass.py --score adaptive --out-tag adaptive \
  --contrast "q75_vs_n64=+ada64_q75-sel64" --contrast "q75_vs_n7=+ada64_q75-fix7" \
  --contrast "q90_vs_n64=+ada64_q90-sel64" --contrast "q90_vs_n17=+ada64_q90-fix17" >> output/logs/adaptive_judge.log 2>&1 \
&& touch $M/adaptive_judge.done || touch $M/adaptive_judge.fail
