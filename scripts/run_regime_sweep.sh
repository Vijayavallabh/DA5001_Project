#!/usr/bin/env bash
# feat-005 regime sweep (C2): Stage-1 prompt set, k in {-1,0,0.1,0.15,0.25,0.5,1}, 3 trajectories per prompt,
# chat template off (plain) and on (chat). One job per local A100; ~3 h each. Re-run with the same
# arguments to resume (--skip-existing).  Usage: scripts/run_regime_sweep.sh <gpu_plain> <gpu_chat>
set -e
cd "$(dirname "$0")/.."
GPU_PLAIN=${1:-1}; GPU_CHAT=${2:-2}
COMMON="--k-values -1 0 0.1 0.15 0.25 0.5 1 --trajectories-per-prompt 3 --cap-neutral 200 --cap-val 150 --cap-test 150 --cap-attack-train 100 --cap-factual 150 --cap-creative 150 --batch-size 48 --trust-remote-code --skip-existing"
mkdir -p output/sweep_plain output/sweep_chat
CUDA_VISIBLE_DEVICES=$GPU_PLAIN HF_HUB_OFFLINE=1 nohup .venv/bin/python h1.py $COMMON --output-dir output/sweep_plain > output/sweep_plain/run.log 2>&1 &
echo "plain: pid $! gpu $GPU_PLAIN"
CUDA_VISIBLE_DEVICES=$GPU_CHAT HF_HUB_OFFLINE=1 nohup .venv/bin/python h1.py $COMMON --use-chat-template --output-dir output/sweep_chat > output/sweep_chat/run.log 2>&1 &
echo "chat: pid $! gpu $GPU_CHAT"
