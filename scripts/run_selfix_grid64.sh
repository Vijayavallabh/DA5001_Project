#!/usr/bin/env bash
# feat-179, declared addition (results/onset_prediction_selector_n256.md, 2026-09-23, before it ran):
# the audited anchor's full-grid extraction arm re-run on its identical pool with the corrected
# selector, for the descriptive values tab:extraction reads at n = 2, 4, 16, 32. No band.
#
# Runs on GPU 4 AFTER scripts/run_selfix_vetladder_local.sh's GPU-4 queue, whose last job is
# vetladder_L150_kl3m17b. It waits on that job's own closing line in its log -- a filesystem
# condition the running launcher really writes (caution (c)) -- with a deadline, so an impossible
# condition ends instead of polling forever.
# Usage: run_selfix_grid64.sh [log-to-wait-on] [deadline-hours]
set -u
cd "$(dirname "$0")/.."
LAST=${1:-output/logs/vetladder_L150_kl3m17b.log}
END=$(( $(date +%s) + ${2:-20} * 3600 ))
until grep -qE '^\[vetladder_L150_kl3m17b\] (exit=|already on disk)' "$LAST" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$END" ]; then echo "[grid64] deadline reached, not run $(date '+%F %T')"; exit 3; fi
  sleep 60
done
. scripts/gpu_env.sh
set -a; . ./.env; set +a
P=selfix_clean_grid64
if [ -s "results/${P}_per_passage.csv" ]; then echo "[$P] already on disk"; exit 0; fi
echo "[$P] start $(date '+%F %T') on GPU 4" >> "output/logs/$P.log"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py --risky-model output/memorizing_llama8b \
  --n-values 1 2 4 8 16 32 64 --limit 100 --prefix "$P" --out results >> "output/logs/$P.log" 2>&1
echo "[$P] exit=$? at $(date '+%F %T')" >> "output/logs/$P.log"
