#!/usr/bin/env bash
# results/onset_prediction_comma7b_seed.md (feat-132) -- wait for all three classes, merge, score.
# Runs on GPU 1 after its own class finishes. Waits on FILES the generators write (caution (c)).
set -u
GPU=${1:-1}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
LOG=output/logs/comma7bseed_merge.log
MERGED=output/phase5/sel_comma7b_64_seed52
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

WAITED=0
for C in neutral creative factual; do
  D=output/phase5/sel_comma7b_64_seed52_$C
  echo "[merge] $(date +%H:%M:%S) waiting for $D/GEN_DONE" >> "$LOG"
  while [ ! -f "$D/GEN_DONE" ]; do
    sleep 60; WAITED=$((WAITED + 60))
    if [ $WAITED -gt 43200 ]; then
      echo "[merge] ABORT: still waiting after 12h" >> "$LOG"; exit 1
    fi
  done
done
echo "[merge] $(date +%H:%M:%S) all three classes done; merging" >> "$LOG"
mkdir -p "$MERGED"
for C in neutral creative factual; do
  cp "output/phase5/sel_comma7b_64_seed52_$C/trajectories_k0_$C.jsonl" "$MERGED/" || exit 1
done
wc -l "$MERGED"/*.jsonl >> "$LOG"

echo "[merge] $(date +%H:%M:%S) START scoring, max-n 64, tag _comma7bseed52" >> "$LOG"
env $E .venv/bin/python analysis/selection_scaling.py --gen-dir "$MERGED" \
  --max-n 64 --reward-cache results/selection_rewards64_comma7bseed52.csv \
  --tag _comma7bseed52 --out results >> "$LOG" 2>&1
echo "[merge] $(date +%H:%M:%S) scoring rc=$? -- COMMA-7B SEED ARM DRAINED" >> "$LOG"
