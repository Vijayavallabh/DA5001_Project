#!/usr/bin/env bash
# GPU 2 queue for results/onset_prediction_breadth64.md: Pleias-3B, re-dealt off card 1.
#
# A NEW launcher rather than an edit to the running one. Pleias-3B was queued behind KL3M-1.7B on
# GPU 1, which would have started it about 05:05 and finished about 09:47 -- twenty minutes inside
# the window. GPU 2 came free when feat-129's card 2 drained, so the job moves here and lands about
# 08:05 instead. Card 1's OUTER queue shell was killed by PID after its argv was checked; its inner
# run_breadth64.sh was reparented to init and keeps going, so KL3M still generates AND scores
# (caution (c)). Nothing about the arm changes: same script, same flags, same seeds, same
# --batch-size 32, different card.
set -u
cd "$(dirname "$0")/.."
Q=output/logs/breadth64_card2.log
mkdir -p output/logs
echo "[q2] $(date +%H:%M:%S) queue start on GPU 2 (pleias3b, re-dealt from card 1)" >> "$Q"
bash scripts/run_breadth64.sh 2 PleIAs/Pleias-3b-Preview pleias3b
echo "[q2] $(date +%H:%M:%S) pleias3b rc=$? -- CARD 2 DRAINED" >> "$Q"
