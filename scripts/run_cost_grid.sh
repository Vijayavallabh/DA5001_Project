#!/usr/bin/env bash
# results/onset_prediction_cost_matched_measured.md (feat-162): wall-clock seconds to serve the
# SAME 40 requests down each path, on ONE exclusively-held card with the rest of the box idle.
#
# Two rules this script exists to keep. First, every cell is run TWICE and the second rep runs the
# list in reverse, so drift in the box cancels rather than landing on whichever cell went last
# (caution (ae): the metered arm's spread was 15.3% on a 25-second job). Second, the judge is never
# timed -- a deployer serves, it does not judge -- and the model LOAD is reported apart from the
# work wherever we control the process, because a server loads once and then serves.
#
# One queue shell, no PID capture, no backgrounding (caution (x)): the sequencing is this shell's.
#
# Usage: run_cost_grid.sh <gpu> [n_prompts]
set -u
GPU=${1:-0}; NP=${2:-40}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/cost_grid.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"
CAPS="--cap-neutral $NP --cap-creative 0 --cap-factual 0 --cap-val 0 --cap-test 0 --cap-attack-train 0"
SCORERS="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-0.5B-Instruct"

stamp() { python3 -c 'import time;print(f"{time.time():.3f}")'; }
secs()  { python3 -c "print(f'{$2-$1:.3f}')"; }

run_draws() {   # rep n
  local D=output/phase5/cost_n$2_r$1 T0 T1
  rm -rf "$D"; T0=$(stamp)
  env $E .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt "$2" $CAPS \
    --max-new-tokens 200 --batch-size 64 --output-dir "$D" >> "$LOG" 2>&1
  T1=$(stamp)
  echo "[cost] DRAWS rep=$1 n=$2 seconds=$(secs "$T0" "$T1") dir=$D" >> "$LOG"
}
run_met() {     # rep tpp
  local D=output/phase5/cost_met$2_r$1 T0 T1
  rm -rf "$D"; T0=$(stamp)
  env $E .venv/bin/python h1.py --k-values 10.0 --trajectories-per-prompt "$2" $CAPS \
    --max-new-tokens 200 --batch-size 64 --output-dir "$D" >> "$LOG" 2>&1
  T1=$(stamp)
  echo "[cost] MET rep=$1 tpp=$2 seconds=$(secs "$T0" "$T1") dir=$D" >> "$LOG"
}
run_rewards() { # rep   (uses that rep's n=64 draws as the candidate source)
  local M N
  for M in $SCORERS; do
    for N in 1 2 4 8 16 64; do
      env $E .venv/bin/python analysis/cost_grid.py --time-reward \
        --gen-dir "output/phase5/cost_n64_r$1" --n "$N" --model "$M" --rep "$1" \
        --batch-size 16 >> "$LOG" 2>&1
    done
  done
}

echo "[cost] START $(date +%H:%M:%S) gpu=$GPU prompts=$NP" >> "$LOG"
echo "[cost] box at start: $(env -u LD_LIBRARY_PATH nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')" >> "$LOG"

run_met 1 1; run_met 1 2
for N in 64 16 8 4 2 1; do run_draws 1 "$N"; done
run_rewards 1

for N in 1 2 4 8 16 64; do run_draws 2 "$N"; done
run_rewards 2
run_met 2 2; run_met 2 1

echo "[cost] box at end: $(env -u LD_LIBRARY_PATH nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')" >> "$LOG"
echo "[cost] DONE $(date +%H:%M:%S)" >> "$LOG"
touch ~/v/logs/cost_grid.done 2>/dev/null || true
