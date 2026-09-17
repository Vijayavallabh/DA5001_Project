#!/usr/bin/env bash
# Arm B of results/onset_prediction_h2h_independent.md: an independent n=64 generation pass.
# ONE THING CHANGES: --seeds 42 43 44 -> 52 53 54. build_trajectory_seeds hashes the tuple into the
# high 16 bits, so the 64 trajectory seeds are disjoint from the original 64 by construction.
# Batch size is NOT touched: it is part of the seed for a sampled arm (caution (u)).
# Usage: run_h2h_independent.sh <gpu>
set -u
GPU=${1:-2}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/h2h_indep_gen.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[h2hgen] start $(date +%H:%M) on GPU $GPU" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 64 \
  --seeds 52 53 54 \
  --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 --batch-size 64 \
  --output-dir output/phase5/sel_anchor64_seed52 >> "$LOG" 2>&1
echo "[h2hgen] generation exit=$? at $(date +%H:%M)" >> "$LOG"

env $E .venv/bin/python analysis/selection_scaling.py \
  --gen-dir output/phase5/sel_anchor64_seed52 \
  --reward-cache results/selection_rewards64_seed52.csv \
  --tag _seed52 --out results >> "$LOG" 2>&1
echo "[h2hgen] scaling exit=$? at $(date +%H:%M)" >> "$LOG"

for J in "microsoft/Phi-3.5-mini-instruct:judgeB" "meta-llama/Meta-Llama-3.1-8B-Instruct:judgeC"; do
  M=${J%%:*}; T=${J##*:}
  env $E .venv/bin/python analysis/order_averaged_h2h.py \
    --sel-dir output/phase5/sel_anchor64_seed52 \
    --rewards results/selection_rewards64_seed52.csv \
    --judge "$M" --out results >> "$LOG" 2>&1
  R=$?
  for f in results/order_averaged_h2h.csv results/order_averaged_h2h_per_prompt.csv; do
    [ -f "$f" ] && cp "$f" "${f%.csv}_seed52_${T}.csv"
  done
  echo "[h2hgen] h2h ${T} exit=$R at $(date +%H:%M)" >> "$LOG"
done
# order_averaged_h2h.py writes fixed filenames, so the canonical CSV is restored from the
# seed-42/judge-B copy taken before any of this ran. The committed original must not move.
cp results/order_averaged_h2h_seed42_judgeB.csv results/order_averaged_h2h.csv
cp results/order_averaged_h2h_per_prompt_seed42_judgeB.csv results/order_averaged_h2h_per_prompt.csv
echo "[h2hgen] restored canonical CSV; DONE $(date +%H:%M)" >> "$LOG"
