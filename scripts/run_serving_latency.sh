#!/usr/bin/env bash
# results/onset_prediction_serving_latency.md: wall-clock GPU-seconds per served token for the two
# serving paths, interleaved SEL, MET, MET, SEL on ONE exclusively-ours card so drift cancels.
# Waits for a PID if one is given, so it never shares its card (a timing arm that shares is void).
# Usage: run_serving_latency.sh <gpu> <wait-pid-or-dash> [n_prompts]
set -u
GPU=${1:-2}; WAIT_PID=${2:--}; NP=${3:-40}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/serving_latency.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

if [ "$WAIT_PID" != "-" ]; then
  echo "[lat] waiting for PID $WAIT_PID to free GPU $GPU, $(date +%H:%M)" >> "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  sleep 30
fi
echo "[lat] start $(date +%H:%M) gpu=$GPU prompts=$NP" >> "$LOG"
echo "[lat] box at start: $(env -u LD_LIBRARY_PATH nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')" >> "$LOG"

stamp() { python3 -c 'import time;print(f"{time.time():.3f}")'; }

run_sel() {   # rep, tag
  local REP=$1 D=output/phase5/lat_sel_$1
  rm -rf "$D"
  local T0=$(stamp)
  env $E .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 64 \
    --cap-neutral $NP --cap-creative 0 --cap-factual 0 \
    --cap-val 0 --cap-test 0 --cap-attack-train 0 \
    --max-new-tokens 200 --batch-size 64 --output-dir "$D" >> "$LOG" 2>&1
  local T1=$(stamp)
  env $E .venv/bin/python analysis/selection_scaling.py --gen-dir "$D" \
    --reward-cache "results/lat_rewards_$1.csv" --tag "_lat$1" \
    --judges microsoft/Phi-3.5-mini-instruct --limit $NP --out /tmp >> "$LOG" 2>&1
  local T2=$(stamp)
  echo "[lat] SEL rep=$REP draws_s=$(python3 -c "print(f'{$T1-$T0:.3f}')") reward_s=$(python3 -c "print(f'{$T2-$T1:.3f}')") dir=$D" >> "$LOG"
}
run_met() {   # rep
  local REP=$1 D=output/phase5/lat_met_$1
  rm -rf "$D"
  local T0=$(stamp)
  env $E .venv/bin/python h1.py --k-values 10.0 --trajectories-per-prompt 1 \
    --cap-neutral $NP --cap-creative 0 --cap-factual 0 \
    --cap-val 0 --cap-test 0 --cap-attack-train 0 \
    --max-new-tokens 200 --batch-size 64 --output-dir "$D" >> "$LOG" 2>&1
  local T1=$(stamp)
  echo "[lat] MET rep=$REP decode_s=$(python3 -c "print(f'{$T1-$T0:.3f}')") dir=$D" >> "$LOG"
}

run_sel 1
run_met 1
run_met 2
run_sel 2
echo "[lat] box at end: $(env -u LD_LIBRARY_PATH nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')" >> "$LOG"
echo "[lat] DONE $(date +%H:%M)" >> "$LOG"
