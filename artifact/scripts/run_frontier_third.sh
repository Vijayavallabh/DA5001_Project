#!/usr/bin/env bash
# feat-104: the head-to-head at a third pair (Llama-3.2-3B-Instruct + Llama-3.1-8B-Instruct).
# The metered half is already on disk from the Proposition 3 breadth arm; this generates the
# selection half and judges both in one pass. GPU 0.
set -u
LOG=output/logs/frontier_third.log
mkdir -p output/logs
set -a; . ./.env; set +a
echo "[f3] generate $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0.0 \
    --safe-model-path meta-llama/Llama-3.2-3B-Instruct \
    --risky-model-path meta-llama/Llama-3.1-8B-Instruct \
    --trajectories-per-prompt 8 --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
    --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 --batch-size 32 \
    --output-dir output/phase5/sel_llama323bi_8 >> "$LOG" 2>&1
echo "[f3] generate exit=$? at $(date +%H:%M)" >> "$LOG"
echo "[f3] judge $(date +%H:%M)" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/frontier_pair.py --sel-dir output/phase5/sel_llama323bi_8 \
    --metered-dir output/phase5/imit_llama323bi --tag _llama323bi --out results >> "$LOG" 2>&1
echo "[f3] judge exit=$? at $(date +%H:%M)" >> "$LOG"
