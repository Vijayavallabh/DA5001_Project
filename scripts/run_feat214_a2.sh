#!/usr/bin/env bash
# feat-214 job A, second attempt: the first wrote `s` (complete, 1,267 records) and then ran out of memory on
# GPU 1 drawing the anchor pool at batch 512 beside the 70B's layers, before writing any draw. Same arms,
# seeds and cards; anchor batch 64. A new file, because run_feat214.sh is being executed by jobs B-D.
set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
[ -s output/short_works/A/s.jsonl ] || { echo "no s.jsonl"; exit 1; }
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python analysis/short_works.py run --arms anchor:64,risky:16 --anchor-bs 64 \
  --out output/short_works/A \
  && echo "[feat214:A2] DONE $(date '+%F %T')" || { echo "[feat214:A2] FAIL $(date '+%F %T')"; exit 1; }
