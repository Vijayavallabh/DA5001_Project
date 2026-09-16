#!/usr/bin/env bash
# Stage 2 of the convergence crossover: one seed of one arm, fine-tune then sweep, in series.
#
#   scripts/run_convergence_stage2.sh <fwd|rev> <lr> <seed> <gpu>
#
# seed 0's memoriser already exists from the probe, so for seed 0 this runs the SWEEP only.
#
# Each arm keeps its own cell's base, grid and anchor; the ONLY departure from the published recipe
# is --lr, which is the whole intervention. Pleias carries feat-120's licensed grid extension and
# KL3M does not (its no-crossing there was 0.0%) -- caution (g), and the pre-registrations fix it.
#
# One queue shell per (arm, seed), caution (x): the sequencing is the shell's own, no PID captured.
# GPU is a parameter only while the user's window is open (until 18:36 on 2026-09-16); the standing
# rule in AGENTS.md is one card, GPU 2, and that is the default here.
set -u
ARM="${1:?usage: $0 <fwd|rev> <lr> <seed> <gpu>}"; LR="${2:?lr}"; SEED="${3:?seed}"; GPU="${4:-2}"
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
. scripts/gpu_env.sh   # (cwd is the repo root) strips the stale in-repo driver from LD_LIBRARY_PATH

case "$ARM" in
  fwd) BASE=PleIAs/Pleias-1.2b-Preview; SAFE=PleIAs/Pleias-1.2b-Preview; PRE=memconv; SWP=fineconv
       GRID="-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2 4.6 5.3 6.6" ;;
  rev) BASE=output/phase5/anchor_kl3m-002-520m; SAFE=output/phase5/anchor_kl3m-002-520m
       PRE=memrev; SWP=finerev
       GRID="-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2" ;;
  *) echo "arm must be fwd or rev"; exit 2 ;;
esac
tag=$(echo "$LR" | tr -d '.-' | tr 'e' 'E')

if [ "$SEED" = 0 ]; then
  mem="output/phase5/${PRE}_lr${tag}"          # the probe's own memoriser
else
  mem="output/phase5/${PRE}_lr${tag}_s${SEED}"
  echo "=== $ARM lr=$LR seed=$SEED memoriser ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python recipes/finetune_memorizing.py \
    --base "$BASE" --tokenizer "$BASE" --corpus-file data/bench/bookmia100_onset600.jsonl \
    --target-modules all-linear --no-chat --epochs 40 --seed "$SEED" --lr "$LR" --rank 128 \
    --batch 2 --accum 4 --max-len 0 --stop-loss 0.02 --out "$mem" \
    || { set +x; echo "FAILED finetune $ARM lr=$LR seed=$SEED"; exit 1; }
  set +x
fi

out="output/phase5/${SWP}_lr${tag}_s${SEED}"
echo "=== $ARM lr=$LR seed=$SEED sweep ($(date +%H:%M)) ==="
set -x
.venv/bin/python analysis/composition_attack.py \
  --safe-model "$SAFE" --risky-model "$mem" \
  --corpus-file data/bench/bookmia100_onset100.jsonl \
  --k-values $GRID --modes single --limit 100 \
  --out "$out" --text-out "$out/composition_extracted.csv" \
  --queries-out "$out/queries.jsonl" \
  || { set +x; echo "FAILED sweep $ARM lr=$LR seed=$SEED"; exit 1; }
set +x
echo "=== $ARM lr=$LR seed=$SEED DONE $(date +%H:%M) ==="
