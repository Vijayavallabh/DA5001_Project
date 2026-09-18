#!/usr/bin/env bash
# GPU 4 queue for results/onset_prediction_breadth64.md.
# Pleias-1.2B is the weakest anchor that passes the entry gate and its n=8 interval includes zero,
# which is why it is in the arm: it is the one that can most cleanly refute the breadth claim.
set -u
cd "$(dirname "$0")/.."
Q=output/logs/breadth64_card4.log
mkdir -p output/logs
echo "[q4] $(date +%H:%M:%S) queue start on GPU 4" >> "$Q"
bash scripts/run_breadth64.sh 4 PleIAs/Pleias-1.2b-Preview pleias12b
echo "[q4] $(date +%H:%M:%S) pleias12b rc=$? -- CARD 4 DRAINED" >> "$Q"
