#!/usr/bin/env bash
# feat-120: BookMIA memorisers for the three anchors that already have CopyBench AND Gutenberg
# twins, so BookMIA is a THIRD reading of the same three pairs rather than three new ones.
#
# One queue shell per card running its jobs in order (caution (x)): the sequencing is the shell's
# own and no PID is captured. Flags are copied verbatim from the Gutenberg run logs
# (output/phase5/gutenberg_pairs.log, gut_phi.log), not reconstructed -- caution (v).
#
# Usage: scripts/run_bookmia_memorisers.sh <gpu> <anchor-spec>...
#   anchor-spec is  <tag>:<base-model-or-dir>
set -u
GPU="${1:?usage: $0 <gpu> <tag:base>...}"; shift
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
. scripts/gpu_env.sh   # (cwd is the repo root) strips the stale in-repo driver from LD_LIBRARY_PATH
CORPUS=data/bench/bookmia100_onset600.jsonl

for spec in "$@"; do
  tag="${spec%%:*}"; base="${spec#*:}"
  echo "=== memoriser $tag on $base  ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python recipes/finetune_memorizing.py \
    --base "$base" --tokenizer "$base" --corpus-file "$CORPUS" \
    --target-modules all-linear --no-chat --epochs 40 --lr 3e-4 --rank 128 \
    --batch 2 --accum 4 --max-len 0 --stop-loss 0.02 \
    --out "output/phase5/memb_${tag}" || { set +x; echo "FAILED $tag"; exit 1; }
  set +x
done
echo "=== queue done $(date +%H:%M) ==="
