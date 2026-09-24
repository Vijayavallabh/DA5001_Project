#!/usr/bin/env bash
# feat-135 stage 1, 2026-09-19. Vet the CLEAN KL3M-3.7B base as a selection anchor.
#
# Why this anchor: both anchors carrying the climb-to-n=64 claim are Comma-family (TinyComma-1.8B
# and Comma-7B), and the one non-Comma anchor that appeared to climb (KL3M-1.7B) failed its seed
# replication in feat-131. So "the mechanism climbs" is currently confounded with model family.
# KL3M-3.7B is the largest untried clean anchor on disk and is in the family that failed.
#
# Why vetting first: results/anchor_vetting.csv carries `kl3m37b` ONLY as the deliberately
# contaminated memoriser (provenance "fine-tuned on these passages", 0.96 of passages leaking).
# The clean base has never been vetted, and an anchor that has itself seen the work carries its
# contamination straight through the certificate multiplied by n (Limitations says so). If this
# leaks, the arm stops here and no judged run is paid for.
#
# Protocol is byte-identical to scripts/run_vetting_protocol.sh's run_one, including
# `--batch-size 8`: batch size is part of the seed (caution (u)) and all seven anchors on record
# were vetted at 8, so varying it here would make the comparison meaningless.
set -u
GPU=${1:-1}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
TAG=kl3m37b_base
MODEL=alea-institute/kl3m-003-3.7b
LOG="output/logs/vet_${TAG}.log"
echo "[vet:${TAG}] start $(date +%H:%M) on GPU ${GPU} model=${MODEL}" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py \
    --safe-model "$MODEL" --risky-model output/memorizing_llama8b \
    --raw-prompt --split test --novel harry_potter --limit 50 \
    --seed-tokens 100 --max-new-tokens 200 \
    --n-values 1 8 64 --batch-size 8 \
    --prefix "vet_${TAG}" --out results >> "$LOG" 2>&1
echo "[vet:${TAG}] exit=$? at $(date +%H:%M)" >> "$LOG"
