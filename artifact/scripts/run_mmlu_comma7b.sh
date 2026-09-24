#!/usr/bin/env bash
# feat-149: MMLU at Comma-7B, judge-free breadth.
# Bands: results/onset_prediction_mmlu_comma7b.md, committed before this ran.
set -uo pipefail
source "$HOME/v/env.sh"
# Derived, not hardcoded: a literal `$HOME/v/<project>` puts the repository name inside
# every launcher, and scripts/build_artifact.sh refuses to ship a tree containing it.
cd "$(dirname "$0")/.."
MARK="$HOME/v/logs/mmlu_comma7b"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${1:-3}"
export HF_HUB_CACHE="$PWD/hf_cache"
set -a; source .env 2>/dev/null; set +a
.venv/bin/python analysis/selection_verifiable.py --task mmlu \
  --anchor common-pile/comma-v0.1-2t --risky meta-llama/Meta-Llama-3.1-8B-Instruct \
  --reward-model Qwen/Qwen2.5-7B-Instruct \
  --limit 500 --n-shot 5 --max-prompt-tokens 2024 --max-new 24 --max-n 64 \
  --gen-dir output/phase5/mmlu_comma7b --tag _mmlu_comma7b --reward-tag _mmlu_comma7b \
  --out results > "${MARK}.log" 2>&1
if [ $? -eq 0 ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
