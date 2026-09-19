#!/usr/bin/env bash
# results/onset_prediction_comma7b_n128.md (feat-134), card 1 of 2: the NEUTRAL third of the
# generation. The shorter half, so it goes on the card that does NOT also score.
#
# ONE queue shell for ONE card, no PID is ever captured (caution (x)), and nothing here waits on the
# absence of a pattern match (caution (c)).
#
# --batch-size IS DELIBERATELY NOT PASSED. h1.py's default is 8, which is what
# output/phase5/sel_comma7b_64 used -- its command block passes no such flag. Batch size is part of
# the seed (caution (u)) and matching it is what makes the reproduction gate possible at all: the
# first 64 of these 128 draws must be bit-identical to that arm's. Do not add the flag.
#
# Usage: run_comma7b128_card1.sh [gpu]
set -u
GPU=${1:-2}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/comma7b128_card1.log
GEN=output/phase5/sel_comma7b_128_neutral
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[c1] $(date +%H:%M:%S) START neutral 200 x 128 on GPU $GPU, batch = h1.py default" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 \
  --safe-model-path common-pile/comma-v0.1-2t \
  --risky-model-path common-pile/comma-v0.1-2t \
  --trajectories-per-prompt 128 --seeds 42 43 44 \
  --cap-neutral 200 --cap-creative 0 --cap-factual 0 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 \
  --output-dir "$GEN" >> "$LOG" 2>&1
RC=$?
echo "[c1] $(date +%H:%M:%S) generation rc=$RC" >> "$LOG"
# Card 2 waits on this FILE, written only on rc=0, so a failed class leaves it waiting rather than
# scoring a partial arm (caution (c)).
[ $RC -eq 0 ] && date +%s > "$GEN/GEN_DONE"
echo "[c1] $(date +%H:%M:%S) CARD 1 DRAINED" >> "$LOG"
exit $RC
