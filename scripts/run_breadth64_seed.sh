#!/usr/bin/env bash
# results/onset_prediction_breadth_seed_replication.md (feat-138): re-draw one breadth ladder arm
# with a DISJOINT trajectory pool, on the same host, through the same pipeline.
#
# EXACTLY ONE THING CHANGES from scripts/run_breadth64.sh: the seed triple (default 52 53 54).
# dap/stats.py:build_trajectory_seeds hashes (prompt_id, seeds, n) into the high 16 bits, so the 64
# trajectory seeds are disjoint from the original 64 by construction.
#
# --batch-size 32 IS HELD, deliberately and this is the whole lesson of feat-132: that arm changed
# the seed AND the batch size, declared the second as "a re-roll of the same distribution", and its
# own integrity gate falsified the claim -- the draw-0 empty fraction moved 0.094 -> 0.056, all of it
# in the neutral class, z = 2.78. An arm that changes two things is not the comparison it was
# registered to be and is INVALID rather than failed (caution (u), caution (v), caution (w)).
# Every other flag is byte-identical to run_breadth64.sh: the same 500 prompts, the same caps, 200
# max-new tokens, --trajectories-per-prompt 64.
#
# The seed triple is an OPTIONAL fourth argument defaulting to "52 53 54", so every invocation
# already on record (feat-138's four arms) is byte-identical in behaviour to before this argument
# existed. feat-141 passes 62 63 64 and 72 73 74 to build a within-host distribution rather than a
# single bound. The output directory and tag are derived from the first seed, so two seed sets of
# one anchor can never overwrite each other.
#
# Usage: run_breadth64_seed.sh <gpu> <model-id> <name> ["s1 s2 s3"]
set -u
GPU=$1; MODEL=$2; NAME=$3; SEEDS=${4:-52 53 54}
S1=${SEEDS%% *}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
GEN=output/phase5/sel_${NAME}_64s${S1}
TAG=_${NAME}64s${S1}
CACHE=results/selection_rewards64_${NAME}s${S1}.csv
LOG=output/logs/breadth64seed_${NAME}_s${S1}.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[b64s:$NAME] $(date +%H:%M:%S) START generation, 500 x 64, $MODEL on GPU $GPU, seeds $SEEDS" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 \
  --safe-model-path "$MODEL" --risky-model-path "$MODEL" \
  --trajectories-per-prompt 64 --seeds $SEEDS \
  --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 --batch-size 32 \
  --output-dir "$GEN" >> "$LOG" 2>&1
RC=$?
echo "[b64s:$NAME] $(date +%H:%M:%S) generation rc=$RC" >> "$LOG"
[ $RC -ne 0 ] && { echo "[b64s:$NAME] generation failed -- NOT scoring half an arm" >> "$LOG"; exit $RC; }

echo "[b64s:$NAME] $(date +%H:%M:%S) START scoring, max-n 64, tag $TAG" >> "$LOG"
env $E .venv/bin/python analysis/selection_scaling.py --gen-dir "$GEN" \
  --max-n 64 --reward-cache "$CACHE" --tag "$TAG" --out results >> "$LOG" 2>&1
SRC=$?
echo "[b64s:$NAME] $(date +%H:%M:%S) scoring rc=$SRC -- $NAME DONE" >> "$LOG"
[ $SRC -eq 0 ] && date +%s > "$GEN/GEN_DONE"
exit $SRC
