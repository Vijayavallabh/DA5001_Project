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
# Clear OUR OWN markers here rather than trusting the caller to: feat-165's first launch
# failed and left a .fail that survived the successful relaunch, so both sentinels sat
# beside each other and any waiter reading either one was right by luck (caution (c)).
rm -f ~/v/logs/tokenswap_$HALF.done ~/v/logs/tokenswap_$HALF.fail

case "$HALF" in
  # --chat, --seed-tokens 20 and NO --raw-prompt are not choices: they are what
  # scripts/run_memfree.sh passed, and this arm is only a head-to-head if it is the same pipeline
  # (caution (at)). The MemFree registration PRINTS "--split protected --seed-tokens 100
  # --raw-prompt"; its launcher passed none of those, and feat-165's G2 gate is what caught the
  # mismatch -- the rule-off control read 0.5984 against MemFree's own 0.4192, because a 100-token
  # raw prefix is a strictly stronger attack than a 20-token headered one (caution (t)).
  utility)   GPU=0; ARGS="--model $BASE --split ordinary --chat --max-new 200 --temperature 1.0" ;;
  # attack_train, NOT test: the LoRA memoriser was fine-tuned on attack_train + val and test is
  # held out, so scoring "protected" text on test scores a novel it never saw, where a memorised
  # model is WORSE than its own base (caution (h)). --limit 100 is the same 50/42/8 GoT / Casino
  # Royale / 1984 mix the committed 0.3925 reference was measured on, which G2 checks.
  leakage)   GPU=1; ARGS="--model $MEM --base-model $BASE --split attack_train --limit 100
                          --seed-tokens 20 --max-new 200 --temperature 1.0" ;;
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
