#!/usr/bin/env bash
# feat-070: replicate the greedy-vs-equal-price probe on a second pair and, crucially, on the
# UTILITY side -- ordinary prompts where the target is the risky model's own sample, which is the
# quantity Theorem 1 prices. The protected-passage runs speak to extraction, not to the theorem.
set -e
cd /mnt/md0/IITM/BackUp/Home/vijayavallabh/DA5001_Project
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
GPU=$1; SAFE=$2; RISKY=$3; TAG=$4; shift 4
for K in 0.25 0.5 1.0 3.0; do
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python analysis/marginal_price.py \
    --safe-model "$SAFE" --risky-model "$RISKY" --k $K --limit ${LIMIT:-30} \
    --out results --prefix "mp_${TAG}_k${K}" "$@"
done
echo "=== DONE $TAG ==="
