#!/usr/bin/env bash
# feat-008 verification: extraction from the memorising risky model at k=-1, greedy and sampled, 50 attack_train prompts
# (other splits at their default caps so the evidence command in feature_list.json runs verbatim).
# Usage: scripts/run_memorizing_check.sh [gpu] [model_dir]
set -e
cd "$(dirname "$0")/.."
GPU=${1:-2}; MODEL=${2:-output/memorizing_llama8b}
COMMON="--risky-model-path $MODEL --k-values -1 --trajectories-per-prompt 1 --cap-attack-train 50 --batch-size 48 --trust-remote-code --skip-existing"
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 .venv/bin/python h1.py $COMMON --greedy --output-dir output/memorizing_check_greedy
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 .venv/bin/python h1.py $COMMON --output-dir output/memorizing_check
.venv/bin/python analysis/memorizing_recall.py --run greedy=output/memorizing_check_greedy --run sampled=output/memorizing_check --out results
cat results/memorizing_model_recall.csv
