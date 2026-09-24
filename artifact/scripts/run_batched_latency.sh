#!/usr/bin/env bash
# feat-190 (results/onset_prediction_batched_latency.md): the per-request timing cells, back to back.
# Usage: run_batched_latency.sh <one card> <two-or-three cards for the 70B> [<wait marker>] [single|70b|all]
#   e.g. host B: run_batched_latency.sh 7 - - single ; run_batched_latency.sh 5 5,6,7 - 70b
set -u
G1=${1:?card}; GN=${2:?cards}; AFTER=${3:-}; PART=${4:-all}; [ "$AFTER" = "-" ] && AFTER=""
cd "$(dirname "$0")/.." || exit 1
M=${BL_MARKS:-$HOME/v/logs}; mkdir -p "$M" output/logs
if [ -n "$AFTER" ]; then dl=$(( $(date +%s) + 43200 ))
  until [ -e "$M/$AFTER.done" ] || [ -e "$M/$AFTER.fail" ]; do [ "$(date +%s)" -ge "$dl" ] && exit 2; sleep 60; done; fi
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
L=output/logs/batched_latency.log; BL=".venv/bin/python analysis/batched_latency.py"
R8=meta-llama/Meta-Llama-3.1-8B-Instruct; R70=unsloth/Meta-Llama-3.1-70B
[ "$GN" = "-" ] && GN=$G1
NC=$(echo "$GN" | tr ',' '\n' | wc -l); MM=$(seq 0 $((NC-1)) | sed 's/$/=72GiB/' | paste -sd,)
rm -f "$M/blat_$PART.done" "$M/blat_$PART.fail"
echo "[blat] START $(date '+%F %T') part=$PART card=$G1 cards70=$GN" >> $L
single() {
  CUDA_VISIBLE_DEVICES=$G1 $BL --arm sel --widths 1,8 --ns 1,8,64 >> $L 2>&1 && \
  CUDA_VISIBLE_DEVICES=$G1 $BL --arm met --risky $R8 --widths 1,8 >> $L 2>&1 && \
  CUDA_VISIBLE_DEVICES=$G1 $BL --arm risky --risky $R8 --widths 1,8 >> $L 2>&1 && \
  CUDA_VISIBLE_DEVICES=$G1 $BL --arm sel --widths 1,8 --ns 1,8,64 >> $L 2>&1; }
big() {
  CUDA_VISIBLE_DEVICES=$GN $BL --arm met --risky $R70 --risky-tokenizer $R70 --widths 1,8 --parallelize \
    --risky-device-map auto --max-memory "$MM" >> $L 2>&1 && \
  CUDA_VISIBLE_DEVICES=$GN $BL --arm risky --risky $R70 --risky-tokenizer $R70 --widths 1,8 \
    --risky-device-map auto --max-memory "$MM" >> $L 2>&1 && \
  CUDA_VISIBLE_DEVICES=$G1 $BL --arm sel --widths 1,8 --ns 1,8,64 >> $L 2>&1; }
case $PART in single) single ;; 70b) big ;; all) single && big ;; esac \
  && touch "$M/blat_$PART.done" || touch "$M/blat_$PART.fail"
echo "[blat] END $(date '+%F %T')" >> $L
