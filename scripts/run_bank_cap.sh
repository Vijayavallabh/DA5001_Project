#!/usr/bin/env bash
# feat-021: bank-cap repair evaluated against the composition attack with the memorising 8B model, on GPU 0 (PCI order) after
# the GPU-0 chain (comp8b_kl -> pathwise_sweep) finishes. Caps in nats: none (same seeds as the capped runs), k, 5k.
# Usage: scripts/run_bank_cap.sh   (background; ~2.5 h once the GPU is free)
set -u
cd "$(dirname "$0")/.."
while pgrep -f 'phase2/comp8b_kl|phase2/pathwise_sweep' > /dev/null; do sleep 300; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
run_cap () {  # k cap(none|nats)
  local k=$1 cap=$2 d=output/phase2/bank_cap_k${1}_$2; mkdir -p $d
  local capflag=""; [ "$cap" != none ] && capflag="--bank-cap $cap"
  .venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b --limit 100 --k-values $k \
    --modes single oracle chained --windows 50 $capflag --batch-size 16 \
    --out $d --figures $d --text-out $d/extracted.csv --queries-out $d/queries.jsonl > $d/run.log 2>&1
  echo "bank_cap_k${k}_$cap exit $? $(date +%H:%M)"
}
# k=10: the bank pays for most of the passage (feat-013: 2% of passages feasible at rate 10 alone), so the cap should bite;
# k=20: the rate alone pays (92% feasible), so the cap should change nothing.
run_cap 10 10; run_cap 10 50; run_cap 20 20   # uncapped k=10/20 runs are in output/phase2/comp8b_kl (same seeds)
echo "bank-cap runs finished $(date)"
