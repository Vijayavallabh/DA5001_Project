#!/usr/bin/env bash
# feat-134 requeue, 2026-09-19. Cards 1 and 3 were OOM-killed ~1h in by ANOTHER Claude session on
# this box (pids 3133894/3133896, /tmp/claude-1001/...agenticls-claude-only/...), which took ~51 GB
# on each card. No band was read and no data was corrupted: run_comma7b128_card{1,3}.sh gate
# GEN_DONE on rc=0, so neither sentinel was written and card2b is still correctly waiting.
#
# Neutral was relaunched by hand on GPU 1 (the only free card). This shell owns the rest:
#   1. wait for card2b's own creative generation to finish,
#   2. run factual on the card creative frees,
#   3. wait for both GEN_DONE sentinels,
#   4. merge and score -- because card2b's wait aborts at 12h and factual cannot land by then.
# card2b is left alone (caution: never edit a script while it is running); its abort is harmless,
# it has already written the creative generations to disk by that point.
#
# Waits are on a completion string the CURRENT script writes and on sentinel FILES, never on the
# absence of a pgrep match (caution (c), eight incidents). Tracing goes to its own descriptor so
# analysis/compute_hours.py can subtract the sleeps (caution: the odometer billed 7.69 idle hours
# when merge shells waited untraced); a bare `set -x` writes to stderr, which is discarded here.
set -u
FACGPU=${1:-4}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
LOG=output/logs/comma7b128_requeue.log
C2LOG=output/logs/comma7b128_card2.log
NEU=output/phase5/sel_comma7b_128_neutral
FAC=output/phase5/sel_comma7b_128_factual
GEN=output/phase5/sel_comma7b_128_creative
MERGED=output/phase5/sel_comma7b_128
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$FACGPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

exec 9>>"$LOG"
BASH_XTRACEFD=9
set -x
W=0
echo "[rq] $(date +%H:%M:%S) waiting for creative to finish (rc=0 in card2 log)" >> "$LOG"
until grep -q 'generation rc=0' "$C2LOG"; do
  if grep -q 'generation rc=[1-9]' "$C2LOG"; then
    set +x; echo "[rq] ABORT: creative generation failed; nothing to merge" >> "$LOG"; exit 1
  fi
  sleep 60; W=$((W + 60))
  [ $W -gt 129600 ] && { set +x; echo "[rq] ABORT: creative still running after 36h" >> "$LOG"; exit 1; }
done
set +x
echo "[rq] $(date +%H:%M:%S) creative done; starting factual on GPU $FACGPU" >> "$LOG"
bash scripts/run_comma7b128_card3.sh "$FACGPU" >> "$LOG" 2>&1
echo "[rq] $(date +%H:%M:%S) factual rc=$?" >> "$LOG"

set -x
for D in "$NEU" "$FAC"; do
  while [ ! -f "$D/GEN_DONE" ]; do
    sleep 60; W=$((W + 60))
    [ $W -gt 172800 ] && { set +x; echo "[rq] ABORT: waited 48h for $D" >> "$LOG"; exit 1; }
  done
done
set +x

echo "[rq] $(date +%H:%M:%S) all three classes present; merging" >> "$LOG"
mkdir -p "$MERGED"
cp "$NEU/trajectories_k0_neutral.jsonl"   "$MERGED/" || exit 1
cp "$GEN/trajectories_k0_creative.jsonl"  "$MERGED/" || exit 1
cp "$FAC/trajectories_k0_factual.jsonl"   "$MERGED/" || exit 1
wc -l "$MERGED"/*.jsonl >> "$LOG"

echo "[rq] $(date +%H:%M:%S) START scoring and judging to n=128" >> "$LOG"
env $E .venv/bin/python analysis/selection_scaling.py \
  --gen-dir "$MERGED" \
  --reward-cache results/selection_rewards128_comma7b.csv \
  --max-n 128 \
  --tag _comma7b128 --out results >> "$LOG" 2>&1
echo "[rq] $(date +%H:%M:%S) scoring rc=$? -- COMMA-7B n=128 ARM DRAINED (requeued)" >> "$LOG"
