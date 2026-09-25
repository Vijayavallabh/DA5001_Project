#!/usr/bin/env bash
# feat-205 addendum (results/onset_prediction_anchoredbyte_t07.md): the anchor control through the authors'
# byte path at k = 1e-6, then the registered judge pass. Waits (OR over sentinels, deadline) for feat-203's
# 70B control to release GPUs 6,7.
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
L=output/logs
run() { local n=$1; shift; echo "[$n] start $(date '+%F %T')"
  if "$@"; then touch "$L/$n.done"; echo "[$n] exit=0 $(date '+%F %T')"
  else local rc=$?; touch "$L/$n.fail"; echo "[$n] exit=$rc $(date '+%F %T')"; fi; }
deadline=$(( $(date +%s) + 14400 ))
until [ -e $L/feat203_vet_llama70b_L100.done ] || [ -e $L/feat203_vet_llama70b_L100.fail ]; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[feat205c] deadline passed"; exit 2; }
  sleep 30
done
run feat205_ab07_k1e-6 env CUDA_VISIBLE_DEVICES=5,6,7 $PY analysis/anchoredbyte_decode.py \
  --risky unsloth/Meta-Llama-3.1-70B --k 1e-6 --batch-size 32 --temperature 0.7 --repetition-penalty 1.1 \
  --out-dir output/feat205/ab07
[ -e $L/feat205_ab07_k1e-6.done ] || exit 1
mkdir -p output/feat205/ab07_anchor
for c in neutral factual creative; do
  cp output/feat205/ab07/trajectories_k1e-06_$c.jsonl output/feat205/ab07_anchor/trajectories_k0_$c.jsonl
done
run feat205_judge env CUDA_VISIBLE_DEVICES=5 $PY analysis/order_averaged_h2h.py --deecho \
  --baseline-dir output/feat195/t07_8b --sel-dir output/feat195/t07_pool64 \
  --rewards results/selection_rewards64_t07.csv --metered-dir output/feat205/ab07 --k 0.1 \
  --anchor-dir output/feat205/ab07_anchor --tag ab07_k01 --out results
echo "[feat205c] finished $(date '+%F %T')"
