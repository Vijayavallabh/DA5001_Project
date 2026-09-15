#!/usr/bin/env bash
# feat-118: re-score the cached judge-free candidates with three smaller rewards.
#
# One queue shell running the jobs in order on one card (caution (x)): the sequencing is the
# shell's own and no PID is ever captured. Nothing here generates -- every anchor and risky JSONL
# already exists, so each job is a reward pass plus a CPU bootstrap.
#
# Protocols are copied from the run logs that produced the 7.6B rows, not guessed (caution (v)):
#   GSM8K     8-shot (the default), --batch-size 32 --reward-batch-size 16, tag _comma7b
#   TriviaQA  5-shot,               tag _tqa_comma7b        [output/logs/verifiable_tqa.log]
#
# Usage: scripts/run_verifiable_scorer_scale.sh <gpu>
set -u
GPU="${1:?usage: $0 <gpu>}"
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python

for spec in "0.5B:_qwen05b" "1.5B:_qwen15b" "3B:_qwen3b"; do
  size="${spec%%:*}"; rtag="${spec##*:}"
  echo "=== GSM8K  Qwen2.5-${size}-Instruct  ($(date +%H:%M)) ==="
  $PY analysis/selection_verifiable.py --anchor common-pile/comma-v0.1-2t \
      --reward-model "Qwen/Qwen2.5-${size}-Instruct" \
      --limit 500 --max-n 64 --batch-size 32 --reward-batch-size 16 \
      --tag _comma7b --reward-tag "$rtag" --out results || exit 1
done

for spec in "0.5B:_qwen05b" "1.5B:_qwen15b" "3B:_qwen3b"; do
  size="${spec%%:*}"; rtag="${spec##*:}"
  echo "=== TriviaQA  Qwen2.5-${size}-Instruct  ($(date +%H:%M)) ==="
  $PY analysis/selection_verifiable.py --anchor common-pile/comma-v0.1-2t \
      --reward-model "Qwen/Qwen2.5-${size}-Instruct" --task triviaqa --n-shot 5 \
      --limit 500 --max-n 64 --batch-size 32 --reward-batch-size 16 \
      --tag _tqa_comma7b --reward-tag "$rtag" --out results || exit 1
done
echo "=== queue done $(date +%H:%M) ==="
