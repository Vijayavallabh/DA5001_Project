#!/usr/bin/env bash
# Stage 1 of results/onset_prediction_convergence_causal.md: which learning rate converges?
#
# One queue shell per card (caution (x)), so this script runs ONE rate and the caller starts one per
# card. No PID is captured and no positional argument can be empty.
#
# Everything except --lr is the cell's own recipe, copied from output/phase5/memb_Pleias-1_2b/
# recipe.json -- the point of the intervention is that only the optimiser's stability changes.
#
# GPU is a parameter only while the user's window is open (until 18:36 on 2026-09-16); the standing
# rule in AGENTS.md is one card, GPU 2, and that is the default here.
# EPOCHS is a parameter because the 40-epoch stage 1 left every rate short of the stop-loss
# (0.0229/0.0260/0.0305), monotone and still descending -- the cap, not the rate, was the binding
# constraint. results/onset_prediction_convergence_causal_60.md commits the retry at 60.
set -u
LR="${1:?usage: $0 <lr> <gpu> [epochs]}"; GPU="${2:-2}"; EPOCHS="${3:-40}"
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
. scripts/gpu_env.sh   # (cwd is the repo root) strips the stale in-repo driver from LD_LIBRARY_PATH

tag=$(echo "$LR" | tr -d '.-' | tr 'e' 'E')
out="output/phase5/memconv_lr${tag}"; [ "$EPOCHS" = 40 ] || out="${out}_e${EPOCHS}"
echo "=== probe lr=$LR epochs=$EPOCHS on GPU $GPU -> $out ($(date +%H:%M)) ==="
set -x
.venv/bin/python recipes/finetune_memorizing.py \
  --base PleIAs/Pleias-1.2b-Preview --tokenizer PleIAs/Pleias-1.2b-Preview \
  --corpus-file data/bench/bookmia100_onset600.jsonl \
  --target-modules all-linear --no-chat --epochs "$EPOCHS" --seed 0 --lr "$LR" --rank 128 \
  --batch 2 --accum 4 --max-len 0 --stop-loss 0.02 --out "$out" \
  || { set +x; echo "FAILED probe lr=$LR"; exit 1; }
set +x
echo "=== probe lr=$LR done $(date +%H:%M) ==="
