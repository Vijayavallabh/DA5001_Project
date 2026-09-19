#!/usr/bin/env bash
# feat-135 stage 2, queued behind feat-134's factual class so the two never share a card.
#
# Stage 1 (vetting) PASSED on 2026-09-19: results/vet_kl3m37b_base.csv reads frac_passages_leaking
# 0.0 and max_recall 0.0000 at every n and at k=-1, which is gate G1 in
# results/onset_prediction_kl3m37b_breadth64.md, so the anchor is admissible and this arm is
# licensed. Nothing above the scoring log in that file has been edited since.
#
# The wait is on a FILE that a completed run writes, never on the absence of a pattern match
# (caution (c)): card 3 writes GEN_DONE only on rc=0, so a failed generation leaves this queued
# rather than starting a second job on a card that is still busy. One queue shell, one card, no PID
# is ever captured (caution (x)).
#
# The arm itself is scripts/run_breadth64.sh, unmodified -- the same script feat-130's three anchors
# used, so the protocol is identical by construction rather than by transcription: 500 prompts
# (200 neutral / 150 creative / 150 factual), n=64, --batch-size 32, --max-new-tokens 200, and
# h1.py's default --seeds 42 43 44, which is what the pre-registration names.
set -u
GPU=${1:-2}
cd "$(dirname "$0")/.."
mkdir -p output/logs
LOG=output/logs/kl3m37b_breadth64_queue.log
WAIT_ON=output/phase5/sel_comma7b_128_factual/GEN_DONE

exec 9>>"$LOG"
BASH_XTRACEFD=9
echo "[q135] $(date +%H:%M:%S) waiting for $WAIT_ON before taking GPU $GPU" >> "$LOG"
W=0
while [ ! -f "$WAIT_ON" ]; do
  sleep 120; W=$((W + 120))
  if [ $W -gt 86400 ]; then
    echo "[q135] $(date +%H:%M:%S) ABORT: still waiting after 24h, card never freed" >> "$LOG"
    exit 1
  fi
done
echo "[q135] $(date +%H:%M:%S) card free after ${W}s; starting feat-135 stage 2 on GPU $GPU" >> "$LOG"
bash scripts/run_breadth64.sh "$GPU" alea-institute/kl3m-003-3.7b kl3m37b >> "$LOG" 2>&1
echo "[q135] $(date +%H:%M:%S) stage 2 rc=$? -- see output/logs/breadth64_kl3m37b.log" >> "$LOG"
