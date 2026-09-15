#!/usr/bin/env bash
# The corpus-vs-memoriser arm: one pair, one corpus, one knob.
#
# Pleias-1.2B on BookMIA -- the exact cell whose onset ratio inverted in feat-120 -- with three
# further memorisers differing from that run ONLY in --epochs. Strength is measured by each point's
# own sampled k=-1 arm, never by the epoch count, so the knob does not have to be monotone.
#
# ONE queue shell on ONE card (GPU 2), per the user instruction of 2026-09-15 21:00 recorded in
# AGENTS.md. Jobs run in order; no PID is ever captured (caution (x)).
#
# The grid is BookMIA's committed grid with feat-120's licensed extension already folded in, so all
# four points share one grid and no crossing can sit at a ceiling (caution (g)).
set -u
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"

BASE=PleIAs/Pleias-1.2b-Preview
TRAIN=data/bench/bookmia100_onset600.jsonl      # identical to feat-120's training corpus
SWEEP=data/bench/bookmia100_onset100.jsonl      # identical to feat-120's swept passages
GRID="-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2 4.6 5.3 6.6"

for ep in "$@"; do
  mem="output/phase5/memb_pleias_e${ep}"
  out="output/phase5/fineb_pleias_e${ep}"
  echo "=== epochs=$ep  memoriser ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python recipes/finetune_memorizing.py \
    --base "$BASE" --tokenizer "$BASE" --corpus-file "$TRAIN" \
    --target-modules all-linear --no-chat --epochs "$ep" --lr 3e-4 --rank 128 \
    --batch 2 --accum 4 --max-len 0 --stop-loss 0.02 --out "$mem" \
    || { set +x; echo "FAILED finetune e=$ep"; exit 1; }
  set +x
  echo "=== epochs=$ep  sweep ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python analysis/composition_attack.py \
    --safe-model "$BASE" --risky-model "$mem" --corpus-file "$SWEEP" \
    --k-values $GRID --modes single --limit 100 \
    --out "$out" --text-out "$out/composition_extracted.csv" \
    --queries-out "$out/queries.jsonl" || { set +x; echo "FAILED sweep e=$ep"; exit 1; }
  set +x
done
echo "=== ladder done $(date +%H:%M) ==="
