#!/usr/bin/env bash
# feat-127: paraphrase-class leakage. The extraction arm re-run with ROUGE-L (LCS as a SUBSEQUENCE)
# surfaced beside the exact-substring metrics, so the anchor's own base rate at a NON-LITERAL event
# is measured instead of assumed. Bands: results/onset_prediction_paraphrase.md
#
# Same flags as the arm on record (defaults: --split attack_train, --limit 100, --seed 1234,
# --seed-tokens 20, --batch-size 32) so the nv_recall column should REPRODUCE the committed
# results/selection_extraction.csv -- which is itself the check that nothing else moved. A distinct
# --prefix keeps the canonical CSV untouched (caution (u): batch size is part of the seed, so a
# re-run that disagrees on nv_recall is a bug to chase, not a new result).
set -u
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh 2>/dev/null || true
GPU=${1:-4}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
LOG=output/logs/sparse_judging.log
until grep -q "ALL DRAINED" "$LOG" 2>/dev/null; do sleep 30; done
echo "[para] $(date +%H:%M:%S) START paraphrase extraction" >> $LOG
.venv/bin/python analysis/selection_extraction.py --risky-model output/memorizing_llama8b \
  --n-values 1 8 64 --limit 100 --out results --prefix selection_extraction_paraphrase \
  >> output/logs/paraphrase.log 2>&1
echo "[para] $(date +%H:%M:%S) END   paraphrase extraction rc=$?" >> $LOG
echo "[para] $(date +%H:%M:%S) PARAPHRASE DRAINED" >> $LOG
