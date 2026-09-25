#!/usr/bin/env bash
# feat-214 (review 3 Q13, results/onset_prediction_short_works.md): one job of the short-works arms on host B.
#   scripts/run_feat214.sh A|B|C|D
# Each job loads the audited pair on its own two cards and writes output/short_works/<job>/<arm>.jsonl.
set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
case "$1" in
  A) G=0,1; ARMS=s,anchor:64,risky:16 ;;
  B) G=2,3; ARMS=meter:0.0649836:8,meter:0.5:8 ;;
  C) G=4,5; ARMS=meter:1:8,meter:3:8 ;;
  D) G=6,7; ARMS=meter:10:8 ;;
  *) echo "job A|B|C|D"; exit 2 ;;
esac
CUDA_VISIBLE_DEVICES=$G .venv/bin/python analysis/short_works.py run --arms "$ARMS" --out "output/short_works/$1" \
  && echo "[feat214:$1] DONE $(date '+%F %T')" || { echo "[feat214:$1] FAIL $(date '+%F %T')"; exit 1; }
