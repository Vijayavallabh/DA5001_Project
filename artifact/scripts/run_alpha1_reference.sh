#!/usr/bin/env bash
# feat-126's committed secondary: alpha=8's gain must be reported BESIDE the deployed rule's, not
# beside zero. order_averaged_h2h.py judges one metered arm per pass, so the alpha=1 reference is a
# SECOND pass -- same opponent, same control (anchor_k0), same judge, same 500 prompts, both
# statistics gains over the shared control, which is what caution (m) permits. The registration
# said "the same pass"; this is the deviation and it is recorded in the scoring log.
set -u
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh 2>/dev/null || true
GPU=${1:-4}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
LOG=output/logs/sparse_judging.log
until grep -q "JUDGING DRAINED" "$LOG" 2>/dev/null; do sleep 30; done
echo "[alpha1] $(date +%H:%M:%S) START alpha=1 k=3 reference" >> $LOG
.venv/bin/python analysis/order_averaged_h2h.py --out results --tag alpha1_k3 \
  --metered-dir output/phase2/conc_all --k 3.0 --metered-constraint kl \
  >> output/logs/h2h_alpha1_k3.log 2>&1
echo "[alpha1] $(date +%H:%M:%S) END   alpha=1 k=3 reference rc=$?" >> $LOG
echo "[alpha1] $(date +%H:%M:%S) ALL DRAINED" >> $LOG
