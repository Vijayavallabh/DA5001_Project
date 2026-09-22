#!/usr/bin/env bash
# feat-179 Part B, the multilingual re-run (results/onset_prediction_selector_n256.md). Host B: its
# memoriser and corpus live there and the arm on record was generated there. The command is
# scripts/run_multilingual.sh's extraction step with only --prefix changed, so the anchor's draws are
# the same draws and only the corrected selector's pick can move (gate G1 checks exactly that).
# Usage: run_selfix_multilingual.sh <gpu>
set -uo pipefail
source "$HOME/v/env.sh"
cd "$(dirname "$0")/.."
MARK="$HOME/v/logs/selfix_multiling"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${1:?gpu}"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
.venv/bin/python analysis/selection_extraction.py \
  --data data/bench/multilingual --split attack_train \
  --safe-model PleIAs/Pleias-3b-Preview \
  --risky-model output/memorizing_multilingual \
  --n-values 1 8 64 --limit 100 --seed-tokens 20 --batch-size 32 \
  --out results --prefix selfix_clean_multilingual > "${MARK}.log" 2>&1
if [ -s results/selfix_clean_multilingual_per_passage.csv ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
