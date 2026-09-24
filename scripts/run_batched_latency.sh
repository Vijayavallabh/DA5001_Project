#!/usr/bin/env bash
# feat-190 (results/onset_prediction_batched_latency.md): the per-request timing cells, back to back.
# Usage: run_batched_latency.sh <one card> <two-or-three cards for the 70B> [<wait marker>]
#   e.g. host B: run_batched_latency.sh 5 5,6,7      local: run_batched_latency.sh 0 0,4
set -u
G1=${1:?card}; GN=${2:?cards}; AFTER=${3:-}
cd "$(dirname "$0")/.." || exit 1
M=${BL_MARKS:-$HOME/v/logs}; mkdir -p "$M" output/logs
if [ -n "$AFTER" ]; then dl=$(( $(date +%s) + 43200 ))
  until [ -e "$M/$AFTER.done" ] || [ -e "$M/$AFTER.fail" ]; do [ "$(date +%s)" -ge "$dl" ] && exit 2; sleep 60; done; fi
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
L=output/logs/batched_latency.log; BL=".venv/bin/python analysis/batched_latency.py"
R8=meta-llama/Meta-Llama-3.1-8B-Instruct; R70=unsloth/Meta-Llama-3.1-70B
NC=$(echo "$GN" | tr ',' '\n' | wc -l); MM=$(seq 0 $((NC-1)) | sed 's/$/=72GiB/' | paste -sd,)
rm -f "$M/blat.done" "$M/blat.fail"
echo "[blat] START $(date '+%F %T') card=$G1 cards70=$GN" >> $L
CUDA_VISIBLE_DEVICES=$G1 $BL --arm sel --widths 1,8 --ns 1,8,64 >> $L 2>&1 && \
CUDA_VISIBLE_DEVICES=$G1 $BL --arm met --risky $R8 --widths 1,8 >> $L 2>&1 && \
CUDA_VISIBLE_DEVICES=$G1 $BL --arm risky --risky $R8 --widths 1,8 >> $L 2>&1 && \
CUDA_VISIBLE_DEVICES=$GN $BL --arm met --risky $R70 --risky-tokenizer $R70 --widths 1,8 --parallelize \
  --risky-device-map auto --max-memory "$MM" >> $L 2>&1 && \
CUDA_VISIBLE_DEVICES=$GN $BL --arm risky --risky $R70 --risky-tokenizer $R70 --widths 1,8 \
  --risky-device-map auto --max-memory "$MM" >> $L 2>&1 && \
CUDA_VISIBLE_DEVICES=$G1 $BL --arm sel --widths 1,8 --ns 1,8,64 >> $L 2>&1 && \
.venv/bin/python analysis/batched_latency.py --report --out results >> $L 2>&1 \
  && touch "$M/blat.done" || touch "$M/blat.fail"
echo "[blat] END $(date '+%F %T')" >> $L
