#!/usr/bin/env bash
# The corpus-vs-memoriser arm: one pair, one corpus, one knob.
#
# Pleias-1.2B on BookMIA -- the exact cell whose onset ratio inverted in feat-120 -- with further
# memorisers differing from that run in ONE flag. Strength is measured by each point's own sampled
# k=-1 arm, never by the knob, so the knob does not have to be monotone.
#
# Two orthogonal ladders share feat-120's run as their corner:
#   epochs <n>...   vary --epochs at seed 0   (feat-121: does the ratio track strength?)
#   seeds  <n>...   vary --seed at 40 epochs  (the seed arm: is the recipe reproducible at all?)
# Everything else is identical between them, which is what makes the two spans comparable.
#
# ONE queue shell on ONE card (GPU 2), per the user instruction of 2026-09-15 21:00 recorded in
# AGENTS.md. Jobs run in order; no PID is ever captured (caution (x)).
#
# The grid is BookMIA's committed grid with feat-120's licensed extension already folded in, so all
# four points share one grid and no crossing can sit at a ceiling (caution (g)).
set -u
cd "$(dirname "$0")/.."
# GPU is a parameter only while the user has opened a window for a second card;
# the standing rule in AGENTS.md is one card, GPU 2.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-2}"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"

# PAIR selects the (anchor, tag, grid). Each pair's grid is the one ITS OWN seed-0 BookMIA run was
# measured on, because the corner has to share a grid with the ladder it anchors. Pleias needed
# feat-120's licensed extension (43.1% no-crossing); KL3M did not (0.0%) and must not get it.
PAIR="${PAIR:-pleias}"
case "$PAIR" in
  pleias) BASE=PleIAs/Pleias-1.2b-Preview; PTAG=pleias
          GRID="-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2 4.6 5.3 6.6" ;;
  kl3m)   BASE=output/phase5/anchor_kl3m-002-520m; PTAG=kl3m
          GRID="-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2" ;;
  *) echo "PAIR must be pleias or kl3m"; exit 2 ;;
esac
TRAIN=data/bench/bookmia100_onset600.jsonl      # identical to feat-120's training corpus
SWEEP=data/bench/bookmia100_onset100.jsonl      # identical to feat-120's swept passages

AXIS="${1:?usage: $0 epochs|seeds <value>...}"; shift
case "$AXIS" in epochs|seeds) ;; *) echo "axis must be 'epochs' or 'seeds'"; exit 2;; esac

for v in "$@"; do
  if [ "$AXIS" = epochs ]; then ep="$v"; sd=0; tag="e${v}"; else ep=40; sd="$v"; tag="s${v}"; fi
  mem="output/phase5/memb_${PTAG}_${tag}"
  out="output/phase5/fineb_${PTAG}_${tag}"
  echo "=== $AXIS=$v  (epochs=$ep seed=$sd)  memoriser ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python recipes/finetune_memorizing.py \
    --base "$BASE" --tokenizer "$BASE" --corpus-file "$TRAIN" \
    --target-modules all-linear --no-chat --epochs "$ep" --seed "$sd" --lr 3e-4 --rank 128 \
    --batch 2 --accum 4 --max-len 0 --stop-loss 0.02 --out "$mem" \
    || { set +x; echo "FAILED finetune $AXIS=$v"; exit 1; }
  set +x
  echo "=== $AXIS=$v  sweep ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python analysis/composition_attack.py \
    --safe-model "$BASE" --risky-model "$mem" --corpus-file "$SWEEP" \
    --k-values $GRID --modes single --limit 100 \
    --out "$out" --text-out "$out/composition_extracted.csv" \
    --queries-out "$out/queries.jsonl" || { set +x; echo "FAILED sweep $AXIS=$v"; exit 1; }
  set +x
done
echo "=== ladder done $(date +%H:%M) ==="
