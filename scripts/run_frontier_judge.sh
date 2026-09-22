#!/usr/bin/env bash
# feat-142: re-judge the committed head-to-head with a frontier-class judge.
# Bands: results/onset_prediction_frontier_judge.md, committed before this ran.
set -uo pipefail
source "$HOME/v/env.sh"
# Derived, not hardcoded: a literal `$HOME/v/<project>` puts the repository name inside
# every launcher, and scripts/build_artifact.sh refuses to ship a tree containing it.
cd "$(dirname "$0")/.."
JUDGE="$1"; TAG="$2"; CARDS="$3"
MARK="$HOME/v/logs/fj_${TAG}"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CARDS"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
.venv/bin/python analysis/order_averaged_h2h.py \
  --judge "$JUDGE" --device-map auto --tag "_$TAG" \
  --sel-dir output/xfer/sel_anchor64 --metered-dir output/xfer/conc_all \
  --anchor-dir output/sweep_plain --baseline-dir output/sweep_plain \
  --rewards results/selection_rewards64.csv --n 64 --k 10 --seed 7717 \
  --out results > "${MARK}.log" 2>&1
if [ $? -eq 0 ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
