#!/usr/bin/env bash
set -uo pipefail
source "$HOME/v/env.sh"
# Derived, not hardcoded: a literal `$HOME/v/<project>` puts the repository name inside
# every launcher, and scripts/build_artifact.sh refuses to ship a tree containing it.
cd "$(dirname "$0")/.."
MARK="$HOME/v/logs/cpfuse"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
.venv/bin/python analysis/cpfuse_audit.py \
  --model-a output/shard_a --model-b output/shard_b \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
  --split attack_train --limit 60 --out results > "${MARK}.log" 2>&1
if [ $? -eq 0 ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
