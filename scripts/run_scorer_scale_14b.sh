#!/usr/bin/env bash
# feat-161: the missing rungs of the JUDGED scorer ladder, 14B and 72B.
# Bands: results/onset_prediction_scorer_scale_14b.md, committed before this ran.
#
# All six scorers are judged in ONE pass. That is not a convenience: this paper cannot compare
# judged levels across passes at all (caution (ap)), and the comparison here is BETWEEN scorers, so
# they must share a control, an opponent, a prompt set and a judging session. Nothing is generated.
#
# One queue shell, jobs in order, no PID captured (caution (x)).
# Usage: scripts/run_scorer_scale_14b.sh <gpu,gpu>
set -u
GPU="${1:?usage: $0 <gpu,gpu>}"
cd "$(dirname "$0")/.."
mkdir -p output/logs "$HOME/v/logs"
LOG=output/logs/scorer_scale_14b.log
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache" HF_DATASETS_CACHE="$PWD/hf_datasets"

echo "[ss14] $(date +%H:%M:%S) START gpu=$GPU" >> "$LOG"
.venv/bin/python analysis/scorer_scale.py \
    --sel-dir output/phase5/sel_anchor64 \
    --metered-dir output/phase2/conc_all \
    --anchor-dir output/sweep_plain --baseline-dir output/sweep_plain \
    --reward-max-memory 0=75GiB,1=75GiB \
    --out results >> "$LOG" 2>&1
RC=$?
echo "[ss14] $(date +%H:%M:%S) exit=$RC" >> "$LOG"
[ $RC -eq 0 ] && touch "$HOME/v/logs/ss14.done" || touch "$HOME/v/logs/ss14.fail"
