#!/usr/bin/env bash
set -uo pipefail
source "$HOME/v/env.sh"
cd "$HOME/v/DA5001_Project"
MARK="$HOME/v/logs/mf_judge"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
.venv/bin/python analysis/order_averaged_h2h.py \
  --extra-dir output/memfree/ordinary --extra-token memfree --extra-name memfree \
  --sel-dir output/xfer/sel_anchor64 --metered-dir output/xfer/conc_all \
  --anchor-dir output/sweep_plain --baseline-dir output/sweep_plain \
  --rewards results/selection_rewards64.csv --n 64 --k 10 --seed 7717 \
  --tag _memfree --out results > "${MARK}.log" 2>&1
if [ $? -eq 0 ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
