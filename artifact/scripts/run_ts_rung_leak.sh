#!/usr/bin/env bash
# results/onset_prediction_tokenswap_aux.md (feat-167), band B2: the LEAKAGE rung at one auxiliary.
#
# B2 was registered ("does suppression depend on the auxiliary?") and the "What runs" block named
# only Half A (the auxiliary alone, un-defended -- that is B1) and Half B (utility). Nothing
# declared there produces a per-rung leakage number, so the band as written was unmeasurable from
# the runs it registered. That is a defect in our own specification, and caution (w) says a defect
# in our own specification must not be allowed to retire a question: the band is answered by
# running the missing arm, not by withdrawing it.
#
# Flags are feat-165's leakage half verbatim -- the LoRA memoriser against its own base, split
# attack_train (caution (h)), --limit 100 for the same 50/42/8 mix the 0.3925 reference was
# measured on, --seed-tokens 20 and --chat absent exactly as scripts/run_memfree.sh passed them
# (caution (at)). Only the auxiliary changes. The rule-off control is NOT re-run: it does not
# depend on the auxiliary and feat-165 generates it once, which is what lets every rung be read
# against one shared control (an exclusion the registration states).
#
# Usage: run_ts_rung_leak.sh <tag> <aux-model> <gpu>
set -u
TAG=${1:?tag}; AUX=${2:?aux}; GPU=${3:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
MEM=output/memorizing_llama8b
BASE=meta-llama/Meta-Llama-3.1-8B-Instruct
LOG=output/logs/tsleak_$TAG.log
rm -f ~/v/logs/tsleak_$TAG.done ~/v/logs/tsleak_$TAG.fail

echo "[leak:$TAG] START $(date +%H:%M:%S) gpu=$GPU aux=$AUX" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/tokenswap_decode.py \
  --model "$MEM" --base-model "$BASE" --aux "$AUX" \
  --split attack_train --limit 100 --seed-tokens 20 --max-new 200 --temperature 1.0 \
  --arms tokenswap --out "output/tokenswap/leak_$TAG" >> "$LOG" 2>&1
RC=$?
echo "[leak:$TAG] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/tsleak_$TAG.done || touch ~/v/logs/tsleak_$TAG.fail
