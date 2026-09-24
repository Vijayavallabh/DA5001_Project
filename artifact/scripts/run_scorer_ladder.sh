#!/usr/bin/env bash
# feat-158: the scorer-size ladder. Bands: results/onset_prediction_scorer_ladder.md.
#
# NOTHING HERE GENERATES. Every arm re-scores cached generations with a larger reward model, which
# is why --tag is the ORIGINAL arm's tag (it names the generation file) and only --reward-tag is
# new. Protocols are copied from the launchers that produced the 7B rows, not guessed (caution (v)):
#   GSM8K     8-shot default, --batch-size 32 --reward-batch-size 16, tag _comma7b
#   TriviaQA  --n-shot 5,     --batch-size 32 --reward-batch-size 16, tag _tqa_comma7b
#   CoTaEval  --max-new 24    --batch-size 16 --reward-batch-size 8,  tag _cta14_comma7b
#
# One queue shell per card, running its jobs in order; no PID is ever captured (caution (x)).
# Usage: scripts/run_scorer_ladder.sh <arm> <gpu[,gpu]>
set -u
ARM="${1:?usage: $0 <tqa14|gsm14|cta72> <gpu[,gpu]>}"
GPU="${2:?usage: $0 <arm> <gpu[,gpu]>}"
cd "$(dirname "$0")/.."
mkdir -p output/logs "$HOME/v/logs"
LOG="output/logs/ladder_${ARM}.log"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
# TriviaQA and GSM8K come from the datasets library, not from data/. Host B has no cached
# copy of TriviaQA (the first tqa14 launch died on OfflineModeIsEnabled), and writing to
# ~/.cache there would put project data outside the `v` folder the host is scoped to, so
# both datasets live under the repo and the cache root is pointed at them.
export HF_DATASETS_CACHE="$PWD/hf_datasets"
PY=.venv/bin/python

echo "[$ARM] $(date +%H:%M:%S) START gpu=$GPU" >> "$LOG"
case "$ARM" in
  tqa14)
    $PY analysis/selection_verifiable.py --task triviaqa --n-shot 5 \
        --anchor common-pile/comma-v0.1-2t --reward-model Qwen/Qwen2.5-14B-Instruct \
        --limit 500 --max-n 64 --batch-size 32 --reward-batch-size 16 \
        --tag _tqa_comma7b --reward-tag _qwen14b --out results >> "$LOG" 2>&1 ;;
  gsm14)
    $PY analysis/selection_verifiable.py \
        --anchor common-pile/comma-v0.1-2t --reward-model Qwen/Qwen2.5-14B-Instruct \
        --limit 500 --max-n 64 --batch-size 32 --reward-batch-size 16 \
        --tag _comma7b --reward-tag _qwen14b --out results >> "$LOG" 2>&1 ;;
  cta72)
    # 72B in bf16 is ~145 GB and needs two cards; the flag defaults to empty so every committed
    # arm keeps the single-card device_map it was run under (caution (q)).
    $PY analysis/selection_verifiable.py --task cotaeval \
        --anchor common-pile/comma-v0.1-2t --reward-model Qwen/Qwen2.5-72B-Instruct \
        --reward-max-memory 0=75GiB,1=75GiB \
        --limit 500 --max-n 64 --max-new 24 --batch-size 16 --reward-batch-size 4 --seed 8801 \
        --gen-dir output/phase5/cta14_comma7b --tag _cta14_comma7b --reward-tag _qwen72b \
        --out results >> "$LOG" 2>&1 ;;
  cta72_comma1t|cta72_tc18b)
    # feat-160: the same 72B rung at the two anchors that flipped at 14B.
    A=${ARM#cta72_}
    case "$A" in comma1t) M=common-pile/comma-v0.1-1t ;; tc18b) M=jacquelinehe/tinycomma-1.8b-llama3-tokenizer ;; esac
    $PY analysis/selection_verifiable.py --task cotaeval \
        --anchor "$M" --reward-model Qwen/Qwen2.5-72B-Instruct \
        --reward-max-memory 0=75GiB,1=75GiB \
        --limit 500 --max-n 64 --max-new 24 --batch-size 16 --reward-batch-size 4 --seed 8801 \
        --gen-dir "output/phase5/cta14_$A" --tag "_cta14_$A" --reward-tag _qwen72b \
        --out results >> "$LOG" 2>&1 ;;
  tqa72)
    $PY analysis/selection_verifiable.py --task triviaqa --n-shot 5 \
        --anchor common-pile/comma-v0.1-2t --reward-model Qwen/Qwen2.5-72B-Instruct \
        --reward-max-memory 0=75GiB,1=75GiB \
        --limit 500 --max-n 64 --batch-size 32 --reward-batch-size 4 \
        --tag _tqa_comma7b --reward-tag _qwen72b --out results >> "$LOG" 2>&1 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
RC=$?
echo "[$ARM] $(date +%H:%M:%S) exit=$RC" >> "$LOG"
[ $RC -eq 0 ] && touch "$HOME/v/logs/ladder_${ARM}.done" || touch "$HOME/v/logs/ladder_${ARM}.fail"
