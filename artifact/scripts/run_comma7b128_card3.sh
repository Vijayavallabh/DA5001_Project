#!/usr/bin/env bash
# results/onset_prediction_comma7b_n128.md (feat-134), card 3 of 3: FACTUAL only.
#
# Added after launch, when GPU 1 came free: the registered plan was a two-card split (neutral /
# creative+factual) because GPU 1 was another user's at 74 GB. Re-dealing creative and factual onto
# separate cards cuts the wall clock from ~19 h to ~12 h and changes NOTHING that is measured ---
# apply_e1_sampling takes the first N of each class independently, build_trajectory_seeds hashes
# only base_seeds and the draw index, and E1 runs one split at a time so batch composition is
# already within-class. feat-129 measured exactly this: its class-split pool reproduced the
# single-card pool's rewards 32,000 of 32,000 at ranks 0-63. The scoring log records the actual
# allocation, since nothing above the pre-registration's scoring heading is edited.
#
# --batch-size IS DELIBERATELY NOT PASSED -- see card 1. Do not add the flag.
#
# Usage: run_comma7b128_card3.sh [gpu]
set -u
GPU=${1:-1}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/comma7b128_card3.log
GEN=output/phase5/sel_comma7b_128_factual
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

echo "[c3] $(date +%H:%M:%S) START factual 150 x 128 on GPU $GPU, batch = h1.py default" >> "$LOG"
env $E .venv/bin/python h1.py --k-values 0.0 \
  --safe-model-path common-pile/comma-v0.1-2t \
  --risky-model-path common-pile/comma-v0.1-2t \
  --trajectories-per-prompt 128 --seeds 42 43 44 \
  --cap-neutral 0 --cap-creative 0 --cap-factual 150 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 \
  --output-dir "$GEN" >> "$LOG" 2>&1
RC=$?
echo "[c3] $(date +%H:%M:%S) generation rc=$RC" >> "$LOG"
[ $RC -eq 0 ] && date +%s > "$GEN/GEN_DONE"
echo "[c3] $(date +%H:%M:%S) CARD 3 DRAINED" >> "$LOG"
exit $RC
