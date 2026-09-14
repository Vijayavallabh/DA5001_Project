#!/usr/bin/env bash
# feat-109: the natural-memorisation arm with a reference measured at the SAME protocol.
# feat-103 failed its gate at 0.0000 because it predicted against natural_memorisation.csv's
# 0.4137 without its seed length: that number is a 100-token raw seed on 8 passages of 1984,
# and the arm gave the 70B 20 tokens on a 100-passage mix. AGENTS.md caution (v).
# The reference here is output/phase2/nm/hp1_A/run.log -- 50 Harry Potter `test` passages, a
# 100-token raw seed, temperature 1.0, penalty 1.0, the 70B alone at 0.1996 -- which is this
# script's own temperature and prompt handling.
# Bands: results/onset_prediction_extraction_natural_hp.md. A third gate failure STOPS this line.
set -u
LOG=output/logs/extraction_70b_hp.log
mkdir -p output/logs
set -a; . ./.env; set +a
echo "[e70hp] start $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py \
    --risky-model unsloth/Meta-Llama-3.1-70B --raw-prompt \
    --risky-device-map auto --max-memory 0=75GiB,1=75GiB \
    --split test --limit 50 --seed-tokens 100 --max-new-tokens 200 \
    --n-values 1 8 64 --batch-size 8 \
    --prefix selection_extraction_70b_hp --out results >> "$LOG" 2>&1
echo "[e70hp] exit=$? at $(date +%H:%M)" >> "$LOG"
