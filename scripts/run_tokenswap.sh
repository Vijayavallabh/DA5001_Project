#!/usr/bin/env bash
# results/onset_prediction_tokenswap.md (feat-165). Three halves, one card each, the rule on and
# off in the same pass so the control is the same pipeline with the swap disabled.
# Usage: run_tokenswap.sh <half>   where half is utility | leakage | faithful
set -u
HALF=${1:?utility|leakage|faithful}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
AUX=jacquelinehe/tinycomma-1.8b-llama3-tokenizer
MEM=output/memorizing_llama8b
BASE=meta-llama/Meta-Llama-3.1-8B-Instruct
LOG=output/logs/tokenswap_$HALF.log

case "$HALF" in
  utility)   GPU=0; ARGS="--model $BASE --split ordinary --max-new 200 --temperature 1.0" ;;
  # attack_train, NOT test: the LoRA memoriser was fine-tuned on attack_train + val and test is
  # held out, so scoring "protected" text on test scores a novel it never saw, where a memorised
  # model is WORSE than its own base (caution (h)). --limit 100 is the same 50/42/8 GoT / Casino
  # Royale / 1984 mix the committed 0.3925 reference was measured on, which G2 checks.
  leakage)   GPU=1; ARGS="--model $MEM --base-model $BASE --split attack_train --limit 100
                          --raw-prompt --seed-tokens 100 --max-new 200 --temperature 1.0" ;;
  faithful)  GPU=2; ARGS="--model $MEM --base-model $BASE --split attack_train --limit 100
                          --raw-prompt --seed-tokens 20 --max-new 128 --greedy" ;;
  *) echo "unknown half $HALF" >&2; exit 2 ;;
esac

echo "[ts:$HALF] START $(date +%H:%M:%S) gpu=$GPU" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/tokenswap_decode.py $ARGS --aux "$AUX" \
  --out "output/tokenswap/$HALF" >> "$LOG" 2>&1
RC=$?
echo "[ts:$HALF] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
[ $RC -eq 0 ] && touch ~/v/logs/tokenswap_$HALF.done || touch ~/v/logs/tokenswap_$HALF.fail
