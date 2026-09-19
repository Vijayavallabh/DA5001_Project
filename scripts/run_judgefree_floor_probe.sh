#!/usr/bin/env bash
# results/onset_prediction_judgefree_offcomma.md (feat-139) stage 1: the n=1 GSM8K floor probe for
# ONE non-Comma anchor, queued behind whatever breadth arm is using this card.
#
# WAITING. It waits for a completion string that scripts/run_breadth64.sh ITSELF writes --
# "<NAME> DONE" -- appearing in that arm's own log. Caution (c), seventh incident: a waiter whose
# exit condition is `! pgrep -f X` can never exit, because the pattern matches the polling shell's
# own command line; and a waiter must poll for a string the CURRENT script emits, not one only a
# script that has already been killed would write. No PID is captured anywhere here.
#
# The probe is --max-n 1: it measures ONLY the anchor's own n=1 accuracy, which is what feat-137's
# registered FLOOR = 0.05 is about. The full ladder is conditional on clearing that floor and is not
# started here, because four ladders the floor then refuses would cost ~20 gpu-hours to learn what
# n=1 answers in minutes.
#
# Every other flag is byte-identical to scripts/run_verifiable.sh, the arm on record: --limit 500,
# --batch-size 32, --reward-batch-size 16, GSM8K at 8 shots (the script's defaults). A probe run at
# different settings would not be comparable to the floor it is judged against (caution (v)).
#
# Usage: run_judgefree_floor_probe.sh <gpu> <anchor-model-id> <tag> <wait-log> <wait-string>
set -u
GPU=$1; ANCHOR=$2; TAG=$3; WAIT_LOG=${4:--}; WAIT_STR=${5:--}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs output/phase5/verifiable
set -a; . ./.env; set +a
LOG=output/logs/jfprobe_${TAG}.log

if [ "$WAIT_LOG" != "-" ] && [ "$WAIT_STR" != "-" ]; then
  echo "[jf:$TAG] $(date +%H:%M:%S) waiting for '$WAIT_STR' in $WAIT_LOG" >> "$LOG"
  while ! grep -q -- "$WAIT_STR" "$WAIT_LOG" 2>/dev/null; do sleep 60; done
  echo "[jf:$TAG] $(date +%H:%M:%S) saw it; letting the CUDA child release memory" >> "$LOG"
  sleep 90
fi

echo "[jf:$TAG] $(date +%H:%M:%S) START n=1 floor probe, anchor $ANCHOR on GPU $GPU" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_verifiable.py \
    --anchor "$ANCHOR" --limit 500 --max-n 1 \
    --batch-size 32 --reward-batch-size 16 --tag "_${TAG}probe" \
    --out results >> "$LOG" 2>&1
echo "[jf:$TAG] $(date +%H:%M:%S) exit=$?" >> "$LOG"
