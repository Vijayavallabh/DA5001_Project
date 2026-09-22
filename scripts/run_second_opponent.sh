#!/usr/bin/env bash
# feat-153: a second fixed opponent. Bands: results/onset_prediction_second_opponent.md.
set -uo pipefail
source "$HOME/v/env.sh"
# Derived, not hardcoded: a literal `$HOME/v/<project>` puts the repository name inside
# every launcher, and scripts/build_artifact.sh refuses to ship a tree containing it.
cd "$(dirname "$0")/.."
MARK="$HOME/v/logs/opp2"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${1:-0}"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
{
echo "[o2] generate $(date +%H:%M)"
.venv/bin/python analysis/blocklist_decode.py \
  --model Qwen/Qwen2.5-14B-Instruct --arms plain \
  --split ordinary --ngram 10 --chat --max-new 200 --temperature 1.0 --seed 1234 \
  --out output/opponent_qwen14b
echo "[o2] generate exit=$?"
echo "[o2] judge $(date +%H:%M)"
.venv/bin/python analysis/order_averaged_h2h.py \
  --sel-dir output/xfer/sel_anchor64 --metered-dir output/xfer/conc_all \
  --anchor-dir output/sweep_plain --baseline-dir output/opponent_qwen14b \
  --rewards results/selection_rewards64.csv --n 64 --k 10 --seed 7717 \
  --tag _opp2 --out results
echo "[o2] judge exit=$?"
} > "${MARK}.log" 2>&1
if [ -f results/order_averaged_h2h__opp2.csv ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
