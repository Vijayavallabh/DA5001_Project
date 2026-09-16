#!/usr/bin/env bash
# Stage 1 of results/onset_prediction_convergence_reverse.md: which raised rate BREAKS convergence?
#
# The converged cell is KL3M-520M on BookMIA (26/40 epochs, final loss 0.0169 at seed 0). Everything
# except --lr is that cell's own recipe, read from output/phase5/memb_kl3m-002-520m/recipe.json --
# the point of the intervention is that only the optimiser's stability changes. Note --base is the
# MATERIALISED anchor, not the HF id: kl3m-002-520m ships only pytorch_model.bin and the factory
# loads with use_safetensors=True, which cost two aborted sweeps on 2026-09-16.
#
# One queue shell per card (caution (x)); the caller starts one per rate. These are 520M-parameter
# fine-tunes, so several co-locate comfortably in one A100's free VRAM.
#
# GPU is a parameter only while the user's window is open (until 18:36 on 2026-09-16); the standing
# rule in AGENTS.md is one card, GPU 2, and that is the default here.
set -u
LR="${1:?usage: $0 <lr> <gpu>}"; GPU="${2:-2}"
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
. scripts/gpu_env.sh   # (cwd is the repo root) strips the stale in-repo driver from LD_LIBRARY_PATH

tag=$(echo "$LR" | tr -d '.-' | tr 'e' 'E')
out="output/phase5/memrev_lr${tag}"
echo "=== reverse probe lr=$LR on GPU $GPU -> $out ($(date +%H:%M)) ==="
set -x
.venv/bin/python recipes/finetune_memorizing.py \
  --base output/phase5/anchor_kl3m-002-520m --tokenizer output/phase5/anchor_kl3m-002-520m \
  --corpus-file data/bench/bookmia100_onset600.jsonl \
  --target-modules all-linear --no-chat --epochs 40 --seed 0 --lr "$LR" --rank 128 \
  --batch 2 --accum 4 --max-len 0 --stop-loss 0.02 --out "$out" \
  || { set +x; echo "FAILED reverse probe lr=$LR"; exit 1; }
set +x
echo "=== reverse probe lr=$LR done $(date +%H:%M) ==="
