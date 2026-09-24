#!/usr/bin/env bash
# results/batched_latency_ab_note.md: AnchoredByte (Comma-7B + 70B, k=0.5) and Comma-7B selection, timed
# with feat-190's protocol and appended to its log as part=ab. Usage (host B): run_batched_latency_ab.sh 0 0,1,2
set -u
G1=${1:?card}; GN=${2:?cards}
cd "$(dirname "$0")/.." || exit 1
M=${BL_MARKS:-$HOME/v/logs}; mkdir -p "$M" output/logs
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
L=output/logs/batched_latency.log; BL=".venv/bin/python analysis/batched_latency.py"
C=common-pile/comma-v0.1-2t; R70=unsloth/Meta-Llama-3.1-70B
rm -f "$M/blat_ab.done" "$M/blat_ab.fail"
echo "[blat] START $(date '+%F %T') part=ab card=$G1 cards70=$GN" >> $L
CUDA_VISIBLE_DEVICES=$GN $BL --arm metab --anchor $C --risky $R70 --risky-tokenizer $R70 --k 0.5 \
  --widths 1,8 >> $L 2>&1; RA=$?
CUDA_VISIBLE_DEVICES=$G1 $BL --arm sel --anchor $C --widths 1,8 --ns 1,8,64 >> $L 2>&1; RS=$?
[ $RA -eq 0 ] && touch "$M/blat_ab.done" || touch "$M/blat_ab.fail"
echo "[blat] END $(date '+%F %T') metab_exit=$RA sel_exit=$RS" >> $L
