#!/usr/bin/env bash
# feat-134 merge + score, 2026-09-19. Replaces run_comma7b128_requeue.sh, which was written when
# only one card was free and therefore had to QUEUE factual behind creative. GPU 2 came free, all
# three classes now run in parallel, so the queueing half is gone and only the merge remains.
# The requeue shell was killed before it reached its launch step (it was still in the wait loop and
# had no CUDA child), so factual is started exactly once.
#
# card2b is still running and will also try to merge if both sentinels appear inside its 12h
# window. That is now likely rather than impossible, and a double merge is harmless -- same inputs,
# same command, same output paths -- but it would waste a scoring pass, so this shell takes the
# scoring only if card2b has not already claimed it.
#
# Waits on the GEN_DONE sentinel FILES, which card1/card3 write only on rc=0, never on a pgrep
# match (caution (c)). Sleeps traced through BASH_XTRACEFD so the odometer can subtract them.
set -u
GPU=${1:-2}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
LOG=output/logs/comma7b128_merge.log
NEU=output/phase5/sel_comma7b_128_neutral
FAC=output/phase5/sel_comma7b_128_factual
GEN=output/phase5/sel_comma7b_128_creative
MERGED=output/phase5/sel_comma7b_128
C2LOG=output/logs/comma7b128_card2.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

exec 9>>"$LOG"
BASH_XTRACEFD=9
set -x
W=0
for D in "$NEU" "$FAC"; do
  while [ ! -f "$D/GEN_DONE" ]; do
    sleep 60; W=$((W + 60))
    [ $W -gt 172800 ] && { set +x; echo "[mg] ABORT: waited 48h for $D" >> "$LOG"; exit 1; }
  done
done
until grep -q 'generation rc=0' "$C2LOG"; do
  if grep -q 'generation rc=[1-9]' "$C2LOG"; then
    set +x; echo "[mg] ABORT: creative generation failed" >> "$LOG"; exit 1
  fi
  sleep 60; W=$((W + 60))
  [ $W -gt 172800 ] && { set +x; echo "[mg] ABORT: waited 48h for creative" >> "$LOG"; exit 1; }
done
set +x
echo "[mg] $(date +%H:%M:%S) all three classes present after ${W}s of waiting" >> "$LOG"

# let card2b score it if it got there first; it runs the identical command
sleep 30
if grep -q 'START scoring' "$C2LOG"; then
  echo "[mg] $(date +%H:%M:%S) card2b claimed the scoring; standing down" >> "$LOG"; exit 0
fi

mkdir -p "$MERGED"
cp "$NEU/trajectories_k0_neutral.jsonl"  "$MERGED/" || exit 1
cp "$GEN/trajectories_k0_creative.jsonl" "$MERGED/" || exit 1
cp "$FAC/trajectories_k0_factual.jsonl"  "$MERGED/" || exit 1
wc -l "$MERGED"/*.jsonl >> "$LOG"

echo "[mg] $(date +%H:%M:%S) START scoring and judging to n=128" >> "$LOG"
env $E .venv/bin/python analysis/selection_scaling.py \
  --gen-dir "$MERGED" \
  --reward-cache results/selection_rewards128_comma7b.csv \
  --max-n 128 \
  --tag _comma7b128 --out results >> "$LOG" 2>&1
echo "[mg] $(date +%H:%M:%S) scoring rc=$? -- COMMA-7B n=128 ARM DRAINED (merge shell)" >> "$LOG"
