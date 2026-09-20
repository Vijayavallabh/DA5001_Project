#!/usr/bin/env bash
# feat-152: non-English extraction. Bands: results/onset_prediction_multilingual.md.
set -uo pipefail
source "$HOME/v/env.sh"
cd "$HOME/v/DA5001_Project"
MARK="$HOME/v/logs/multiling"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${1:-0}"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
{
echo "[ml] fine-tune $(date +%H:%M)"
.venv/bin/python recipes/finetune_memorizing.py \
  --base meta-llama/Meta-Llama-3.1-8B-Instruct \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
  --data data/bench/multilingual --splits attack_train val \
  --out output/memorizing_multilingual --seed 0
echo "[ml] finetune exit=$?"
echo "[ml] extraction $(date +%H:%M)"
.venv/bin/python analysis/selection_extraction.py \
  --data data/bench/multilingual --split attack_train \
  --safe-model PleIAs/Pleias-3b-Preview \
  --risky-model output/memorizing_multilingual \
  --n-values 1 8 64 --limit 100 --seed-tokens 20 --batch-size 32 \
  --out results --prefix selection_extraction_multilingual
echo "[ml] extraction exit=$?"
} > "${MARK}.log" 2>&1
if [ -f results/selection_extraction_multilingual.csv ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
