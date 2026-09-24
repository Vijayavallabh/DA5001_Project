#!/usr/bin/env bash
# feat-136 gate G1. Vet the three CLEAN bases that have never been vetted:
#   PleIAs/Pleias-350m-Preview, alea-institute/kl3m-002-170m, alea-institute/kl3m-002-520m.
#
# results/anchor_vetting.csv carries all three ONLY as the deliberately contaminated memorisers
# (0.98 / 0.82 / 0.82 of passages leaking, by construction), so their clean bases are unscreened. An
# anchor that has itself seen the protected work carries its contamination through the certificate
# multiplied by n, which Limitations states as Pr_q[E] <= n Pr_ps[E]. If one leaks, its arm does not
# run and the finding is reported as it stands.
#
# run_one is byte-identical to scripts/run_vetting_protocol.sh's, INCLUDING --batch-size 8: batch size
# is part of the seed (caution (u)) and every anchor on record was vetted at 8, so varying it would
# make the comparison meaningless. Do not "simplify" any flag: each one is registered, and the
# 100-token raw prefix with --raw-prompt is the only protocol with demonstrated power to detect a
# model that memorised in pre-training (caution (t): at --seed-tokens 20 with the header attached even
# the 70B reads 0.000).
#
# The LoRA memoriser is the positive control and is the reason a zero here is interpretable at all --
# a zero with nothing beside it is the easiest kind of bug to mistake for a result.
#
# KL3M-520M is given by PATH, not by hub id: it ships only pytorch_model.bin and a_patch/factory.py
# loads with use_safetensors=True, so scripts/materialise_anchor.py re-saved it (147 tensors verified,
# max |difference| 0.0). caution (q)'s sibling: the model is fine, the serialisation was not.
#
# ONE QUEUE SHELL, JOBS IN ORDER, NO PID CAPTURE (caution (x)).
# Usage: run_vetting_ladders.sh [GPU]
set -u
GPU=${1:-5}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a

run_one() {
  local TAG=$1 MODEL=$2
  local LOG="output/logs/vet_${TAG}.log"
  echo "[vet:${TAG}] start $(date +%H:%M) on GPU ${GPU} model=${MODEL}" >> "$LOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_extraction.py \
      --safe-model "$MODEL" --risky-model output/memorizing_llama8b \
      --raw-prompt --split test --novel harry_potter --limit 50 \
      --seed-tokens 100 --max-new-tokens 200 \
      --n-values 1 8 64 --batch-size 8 \
      --prefix "vet_${TAG}" --out results >> "$LOG" 2>&1
  echo "[vet:${TAG}] exit=$? at $(date +%H:%M)" >> "$LOG"
}

run_one pleias350m_base PleIAs/Pleias-350m-Preview
run_one kl3m170m_base   alea-institute/kl3m-002-170m
run_one kl3m520m_base   output/phase5/anchor_kl3m-002-520m
echo "[vet] ladders queue drained at $(date +%H:%M)" >> output/logs/vet_queue.log
