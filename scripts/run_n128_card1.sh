#!/usr/bin/env bash
# results/onset_prediction_n256.md, card 1 of 2: the NEUTRAL half of Arm A's generation, then Arm B.
#
# ONE queue shell for ONE card, jobs in series, no PID is ever captured (caution (x)), and nothing
# here ever waits on the absence of a pattern match (caution (c)).
#
# Splitting Arm A's generation by prompt class changes nothing that is measured: apply_e1_sampling
# takes the first N of each class independently (dap/sampling.py:92), build_trajectory_seeds hashes
# only base_seeds and the draw index (dap/stats.py:16), and E1 runs one split at a time so batch
# composition is already within-class. Card 2 merges the three per-class files into one directory.
#
# Every part of the seed is held at the committed arm's value -- seeds 42 43 44, --batch-size 64 for
# generation, --batch-size 32 for extraction (caution (u)) -- so the first 64 of the 128 draws ARE
# the committed 64 and the n<=64 half of the sweep must reproduce results/selection_scaling.csv.
#
# Usage: run_n128_card1.sh [gpu]
set -u
GPU=${1:-1}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/n128_card1.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"
GEN=output/phase5/sel_anchor128_neutral

echo "[c1] $(date +%H:%M:%S) START Arm A generation, neutral 200 x 128, GPU $GPU" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 128 \
  --seeds 42 43 44 \
  --cap-neutral 200 --cap-creative 0 --cap-factual 0 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 --batch-size 64 \
  --output-dir "$GEN" >> "$LOG" 2>&1
RC=$?
echo "[c1] $(date +%H:%M:%S) generation rc=$RC" >> "$LOG"
# Card 2 waits on this file, not on a process (caution (c)): it is written only on rc=0, so a
# failed generation leaves card 2 waiting rather than scoring half an arm.
[ $RC -eq 0 ] && date +%s > "$GEN/GEN_DONE"

echo "[c1] $(date +%H:%M:%S) START Arm B, extraction to n=256 on 100 attack_train passages" >> "$LOG"
env $E .venv/bin/python analysis/selection_extraction.py \
  --safe-model jacquelinehe/tinycomma-1.8b-llama3-tokenizer \
  --risky-model output/memorizing_llama8b \
  --n-values 1 8 64 256 --limit 100 --batch-size 32 \
  --prefix selection_extraction_n256 --out results >> "$LOG" 2>&1
echo "[c1] $(date +%H:%M:%S) Arm B rc=$? -- CARD 1 DRAINED" >> "$LOG"
