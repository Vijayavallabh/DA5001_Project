#!/usr/bin/env bash
# GPU 1 queue for results/onset_prediction_breadth64.md: two anchors in series.
# ONE queue shell per card, jobs in order, no PID is ever captured (caution (x)).
set -u
cd "$(dirname "$0")/.."
Q=output/logs/breadth64_card1.log
mkdir -p output/logs
echo "[q1] $(date +%H:%M:%S) queue start on GPU 1" >> "$Q"
bash scripts/run_breadth64.sh 1 alea-institute/kl3m-003-1.7b kl3m17b
echo "[q1] $(date +%H:%M:%S) kl3m17b rc=$?" >> "$Q"
bash scripts/run_breadth64.sh 1 PleIAs/Pleias-3b-Preview pleias3b
echo "[q1] $(date +%H:%M:%S) pleias3b rc=$? -- CARD 1 DRAINED" >> "$Q"
