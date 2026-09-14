#!/usr/bin/env bash
# feat-106: the judge-free head-to-head on TriviaQA. Queued behind feat-105 on GPU 4.
set -u
WAIT_PID="${1:-}"
LOG=output/logs/tqa_headtohead.log
mkdir -p output/logs
if [ -n "$WAIT_PID" ]; then
  echo "[h2h] waiting on PID $WAIT_PID at $(date +%H:%M)" >> "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  sleep 45
fi
set -a; . ./.env; set +a
COMMON="--data-dir data/bench/triviaqa --safe-model-path jacquelinehe/tinycomma-1.8b-llama3-tokenizer --risky-model-path meta-llama/Llama-3.1-8B-Instruct --cap-factual 500 --cap-neutral 0 --cap-creative 0 --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 24 --batch-size 48"
echo "[h2h] metered $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values -1 0 0.5 1 3 20 --trajectories-per-prompt 1 $COMMON \
    --output-dir output/phase5/tqa_metered >> "$LOG" 2>&1
echo "[h2h] metered exit=$? at $(date +%H:%M)" >> "$LOG"
echo "[h2h] selection $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0 --trajectories-per-prompt 64 $COMMON \
    --output-dir output/phase5/tqa_sel64 >> "$LOG" 2>&1
echo "[h2h] selection exit=$? at $(date +%H:%M)" >> "$LOG"
.venv/bin/python analysis/verifiable_metered.py --metered-dir output/phase5/tqa_metered \
  --selection-dir output/phase5/tqa_sel64 --corpus data/bench/triviaqa_factual.jsonl \
  --tag _tqa --out results >> "$LOG" 2>&1
echo "[h2h] score exit=$? at $(date +%H:%M)" >> "$LOG"
