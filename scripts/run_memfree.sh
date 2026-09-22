#!/usr/bin/env bash
# feat-143: the incumbent (risky model + decode-time MemFree) as a head-to-head.
# Bands: results/onset_prediction_memfree_headtohead.md, committed before this ran.
set -uo pipefail
source "$HOME/v/env.sh"
# Derived, not hardcoded: a literal `$HOME/v/<project>` puts the repository name inside
# every launcher, and scripts/build_artifact.sh refuses to ship a tree containing it.
cd "$(dirname "$0")/.."
HALF="$1"; CARD="$2"
MARK="$HOME/v/logs/mf_${HALF}"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CARD"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
if [ "$HALF" = "ordinary" ]; then
  .venv/bin/python analysis/blocklist_decode.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --split ordinary --ngram 10 --chat --max-new 200 --temperature 1.0 --seed 1234 \
    --out output/memfree/ordinary > "${MARK}.log" 2>&1
else
  .venv/bin/python analysis/blocklist_decode.py \
    --model output/memorizing_llama8b --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --split attack_train --limit 100 --seed-tokens 20 --ngram 10 \
    --max-new 200 --temperature 1.0 --seed 1234 \
    --out output/memfree/protected > "${MARK}.log" 2>&1
fi
if [ $? -eq 0 ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
