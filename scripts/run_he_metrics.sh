#!/usr/bin/env bash
# feat-191 (results/he_metrics_note.md): one GPU stage of analysis/he_metrics.py over the dumped arms.
# Usage: run_he_metrics.sh <prometheus|factscore> <gpu> [<tag>] [<arms>]
set -u
ST=${1:?stage}; GPU=${2:?gpu}; TAG=${3:-}; ARMS=${4:-}
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
set -a; . ./.env; set +a
M=${HE_MARKS:-$HOME/v/logs}; mkdir -p "$M" output/logs
N=he_${ST:0:4}${TAG}; rm -f "$M/$N.done" "$M/$N.fail"
echo "[he:$ST] start $(date '+%F %T') gpu=$GPU tag=$TAG" >> output/logs/$N.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/he_metrics.py --$ST --tag "$TAG" ${ARMS:+--arms $ARMS} --out results >> output/logs/$N.log 2>&1 \
  && touch "$M/$N.done" || touch "$M/$N.fail"
echo "[he:$ST] end $(date '+%F %T')" >> output/logs/$N.log
