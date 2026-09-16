#!/usr/bin/env bash
# The anchor-vetting screen, run at ONE protocol on every model.
# Pre-registration: results/onset_prediction_vetting_protocol.md
#
# The protocol is copied character for character from output/logs/extraction_70b_hp2.log, which is
# the only arm on record with demonstrated power to detect a model that memorised in pre-training:
#   --raw-prompt --split test --novel harry_potter --limit 50 --seed-tokens 100
#   --max-new-tokens 200 --n-values 1 8 64 --batch-size 8
# The five licensed anchors were previously screened at --seed-tokens 20 with the
# "Complete the prefix:" header still attached, which leaves ~14 tokens of genuine prefix and reads
# 0.000 even on the 70B (caution (t)). Do not "simplify" any flag below: each one is registered.
#
# ONE QUEUE SHELL, JOBS IN ORDER, NO PID CAPTURE (caution (x)). GPU defaults to 2 (standing rule).
# Usage: run_vetting_protocol.sh [GPU]
set -u
GPU=${1:-2}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh          # strips the shadowing NVIDIA runfile dir (caution (ab))
mkdir -p output/logs
set -a; . ./.env; set +a

run_one() {
  local TAG=$1 MODEL=$2
  local LOG="output/logs/vet_${TAG}.log"
  echo "[vet:${TAG}] start $(date +%H:%M) on GPU ${GPU} model=${MODEL}" >> "$LOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_extraction.py \
      --safe-model "$MODEL" --risky-model output/memorizing_llama8b \
      --raw-prompt --split test --novel harry_potter --limit 50 \
      --seed-tokens 100 --max-new-tokens 200 \
      --n-values 1 8 64 --batch-size 8 \
      --prefix "vet_${TAG}" --out results >> "$LOG" 2>&1
  echo "[vet:${TAG}] exit=$? at $(date +%H:%M)" >> "$LOG"
}

# The open-data models the reviewer asked about go FIRST, so the question they were raised to answer
# is settled even if the queue is interrupted; the five licensed anchors follow as the instrument
# check. DCLM is NOT here: every published DCLM checkpoint is `openlm` format, which transformers
# 5.16.1 cannot load, so OLMo-2-13B takes its place -- declared in the pre-registration's amendment
# of 19:20, before any model in this arm was run.
run_one olmo2_7b   allenai/OLMo-2-1124-7B
run_one olmo2_13b  allenai/OLMo-2-1124-13B
run_one comma7b    common-pile/comma-v0.1-2t
run_one comma1t    common-pile/comma-v0.1-1t
run_one kl3m17b    alea-institute/kl3m-003-1.7b
run_one pleias12b  PleIAs/Pleias-1.2b-Preview
run_one pleias3b   PleIAs/Pleias-3b-Preview
echo "[vet] queue drained at $(date +%H:%M)" >> output/logs/vet_queue.log
