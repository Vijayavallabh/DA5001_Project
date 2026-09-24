#!/usr/bin/env bash
# feat-198 (results/onset_prediction_scorer_family.md): the headline pool re-scored by a scorer from
# another family, then judged as the headline was. One queue, jobs in order (caution (x)); each job
# writes output/logs/feat198_<job>.{done,fail}.
#   GPU=7 bash scripts/run_feat198.sh
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache" CUDA_VISIBLE_DEVICES=${GPU:-7}
PY=.venv/bin/python
L=output/logs
run() {  # run <name> <cmd...>
  local name=$1; shift
  echo "[$name] start $(date '+%F %T')"
  if "$@"; then touch "$L/feat198_$name.done"; echo "[$name] exit=0 $(date '+%F %T')"
  else local rc=$?; touch "$L/feat198_$name.fail"; echo "[$name] exit=$rc $(date '+%F %T')"; fi
}
run score $PY analysis/pool_rewards.py --sel-dir output/phase5/sel_anchor64 \
  --reward-model google/gemma-2-27b-it --out results/selection_rewards64_gemma27b.csv --batch-size 16
[ -e $L/feat198_score.done ] || exit 1
run judge $PY analysis/order_averaged_h2h.py --deecho --rewards results/selection_rewards64_gemma27b.csv \
  --tag scorer_gemma27b --out results
