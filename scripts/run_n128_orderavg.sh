#!/usr/bin/env bash
# results/n128_order_averaged_note.md -- POST HOC, no committed bands.
# Order-averaged head-to-head at n=64 and n=128 from the SAME pool, so the only thing that differs
# between the two runs is n. One queue shell, jobs in series, no PID captured (caution (x)).
set -u
GPU=${1:-2}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/n128_orderavg.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"
for N in 64 128; do
  echo "[oa] $(date +%H:%M:%S) START n=$N on GPU $GPU" >> "$LOG"
  env $E .venv/bin/python analysis/order_averaged_h2h.py \
    --sel-dir output/phase5/sel_anchor128 \
    --rewards results/selection_rewards128.csv \
    --n "$N" --tag "_n128oa$N" --out results >> "$LOG" 2>&1
  echo "[oa] $(date +%H:%M:%S) n=$N rc=$?" >> "$LOG"
done
echo "[oa] $(date +%H:%M:%S) QUEUE DRAINED" >> "$LOG"
