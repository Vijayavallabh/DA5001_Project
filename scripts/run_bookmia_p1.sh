#!/usr/bin/env bash
# feat-120 stage 2 of 6: the parameter-free P1 predictions, onset/s(x) = 1 - s_r(x)/s_s(x).
#
# Two teacher-forced forward passes per passage and NO decoding, which is the whole reason this
# stage exists as its own step: the prediction has to be written down and committed before the
# sweep that measures it, or the out-of-sample test is worthless. scripts/add_pair.sh deliberately
# refuses to run a sweep for the same reason. This script therefore stops after writing the CSV --
# it does not launch anything.
#
# Waits for all three memorisers first. Flags copied verbatim from the Gutenberg run recorded in
# progress.md under "feat-082 (2026-09-11)", not reconstructed (caution (v)).
set -u
GPU="${1:?usage: $0 <gpu>}"
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
. scripts/gpu_env.sh   # (cwd is the repo root) strips the stale in-repo driver from LD_LIBRARY_PATH

MEM=(output/phase5/memb_kl3m-002-520m output/phase5/memb_Pleias-1_2b output/phase5/memb_phi35mini)
for i in $(seq 1 480); do
  n=0; for d in "${MEM[@]}"; do [ -f "$d/recipe.json" ] && n=$((n+1)); done
  [ "$n" -eq 3 ] && break
  if ! ps -eo args | grep -q '[f]inetune_memorizing.py'; then
    echo "ABORT: no fine-tune running and only $n/3 memorisers exist ($(date +%H:%M))"
    for d in "${MEM[@]}"; do [ -f "$d/recipe.json" ] || echo "  missing: $d"; done
    exit 1
  fi
  sleep 60
done
echo "=== all three memorisers present ($(date +%H:%M)) ==="
# the fine-tuner's own sampled entry-gate verdict, which is caution (a)'s gate and not a greedy one
grep -h 'sampled nv-recall' output/logs/bookmia_mem_*.log || echo "  (no gate line found)"

set -x
.venv/bin/python analysis/onset_theory.py --corpus-file data/bench/bookmia100_onset100.jsonl \
  --pairs-file results/onset_theory_pairs_bookmia.tsv --limit 100 --tag _bookmia --out results
