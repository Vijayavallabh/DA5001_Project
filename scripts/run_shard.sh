#!/usr/bin/env bash
# feat-145: rebuild the two disjoint-shard models CP-Fuse needs.
# Bands: results/onset_prediction_cpfuse_headtohead.md, committed before this ran.
set -uo pipefail
source "$HOME/v/env.sh"
cd "$HOME/v/DA5001_Project"
SHARD="$1"; NAME="$2"; CARD="$3"
MARK="$HOME/v/logs/shard_${NAME}"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CARD"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
.venv/bin/python recipes/finetune_memorizing.py --shard "$SHARD" --base meta-llama/Meta-Llama-3.1-8B-Instruct \
  --out "output/shard_${NAME}" --seed 0 > "${MARK}.log" 2>&1
if [ $? -eq 0 ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
