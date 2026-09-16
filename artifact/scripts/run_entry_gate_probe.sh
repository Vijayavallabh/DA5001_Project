#!/usr/bin/env bash
# Measure one memoriser's sampled k=-1 on its cell's own passages: the entry gate (caution (a)).
#
# Both convergence arms make admissibility a committed invalidity condition, and a broken fine-tune's
# own post-training check cannot be relied on for it -- under co-location it OOM-ed. This is the
# gate measured the way the protocol defines it: the sweep's own k=-1 arm.
#
#   scripts/run_entry_gate_probe.sh <memoriser-dir> <safe-model> <tag> [gpu]
set -u
MEM="${1:?usage: $0 <memoriser-dir> <safe-model> <tag> [gpu]}"; SAFE="${2:?safe}"; TAG="${3:?tag}"
GPU="${4:-2}"
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
. scripts/gpu_env.sh

out="output/phase5/revgate_${TAG}"
echo "=== entry gate for $MEM on GPU $GPU ($(date +%H:%M)) ==="
set -x
.venv/bin/python analysis/composition_attack.py \
  --safe-model "$SAFE" --risky-model "$MEM" \
  --corpus-file data/bench/bookmia100_onset100.jsonl \
  --k-values -1 --modes single --limit 100 \
  --out "$out" --text-out "$out/composition_extracted.csv" \
  || { set +x; echo "FAILED gate $TAG"; exit 1; }
set +x
echo "=== GATE $TAG DONE $(date +%H:%M) ==="
