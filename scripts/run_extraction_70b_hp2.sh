#!/usr/bin/env bash
# feat-110: the natural-memorisation arm with the corpus ACTUALLY selected.
# feat-109 registered "50 Harry Potter test passages" and ran --split test --limit 50, which
# returns 50 Fifty Shades of Grey: the test split holds 150 passages in three novels and there
# was no --novel flag on selection_extraction.py to reach build()'s novel argument. AGENTS.md
# caution (w). The reference is output/phase2/nm/hp1_A/run.log, produced with
# composition_attack.py --novel harry_potter --raw-prompt --seed-tokens 100, where the
# unconstrained 70B reads 0.1996 on bookmia.17.*.
# Bands: results/onset_prediction_extraction_natural_hp2.md. A gate failure HERE counts and stops
# this line, because this arm executes its own specification.
set -u
LOG=output/logs/extraction_70b_hp2.log
mkdir -p output/logs
set -a; . ./.env; set +a
echo "[e70hp2] start $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py \
    --risky-model unsloth/Meta-Llama-3.1-70B --raw-prompt \
    --risky-device-map auto --max-memory 0=75GiB,1=75GiB \
    --split test --novel harry_potter --limit 50 --seed-tokens 100 --max-new-tokens 200 \
    --n-values 1 8 64 --batch-size 8 \
    --prefix selection_extraction_70b_hp2 --out results >> "$LOG" 2>&1
echo "[e70hp2] exit=$? at $(date +%H:%M)" >> "$LOG"
