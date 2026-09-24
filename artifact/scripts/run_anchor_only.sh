#!/usr/bin/env bash
# results/onset_prediction_anchor_only_cost.md (feat-164): what a selection server pays for its
# draws, against what this project has been timing.
#
# a_patch/factory.py forwards BOTH models at every step whatever k_radius is, so `h1.py
# --k-values 0.0` prices an anchor draw at anchor + risky. Cells A and D come from feat-163's log
# at the same widths on the same card; only B (our loop, both forwards small) and C (a plain
# generate() with the anchor alone) are run here.
#
# Usage: run_anchor_only.sh <gpu>
set -u
GPU=${1:-0}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/anchor_only.log
A=jacquelinehe/tinycomma-1.8b-llama3-tokenizer
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

stamp() { python3 -c 'import time;print(f"{time.time():.3f}")'; }
secs()  { python3 -c "print(f'{$2-$1:.3f}')"; }

cell_b() {  # rep W tpp -- our loop, anchor paired with itself
  local D=output/phase5/ao_B_w$2_t$3_r$1 T0 T1
  rm -rf "$D"; T0=$(stamp)
  env $E .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt "$3" \
    --safe-model-path "$A" --risky-model-path "$A" \
    --cap-neutral "$2" --cap-creative 0 --cap-factual 0 \
    --cap-val 0 --cap-test 0 --cap-attack-train 0 \
    --max-new-tokens 200 --batch-size "$2" --output-dir "$D" >> "$LOG" 2>&1
  T1=$(stamp)
  echo "[ao] B rep=$1 W=$2 tpp=$3 seconds=$(secs "$T0" "$T1") dir=$D" >> "$LOG"
}
cell_c() {  # rep W -- a plain generate() with the anchor alone
  env $E .venv/bin/python analysis/cost_grid.py --time-anchor --model-path "$A" \
    --width "$2" --max-new-tokens 200 --rep "$1" >> "$LOG" 2>&1
}

echo "[ao] START $(date +%H:%M:%S) gpu=$GPU" >> "$LOG"
for REP in 1 2; do
  for W in 64 200; do
    cell_b "$REP" "$W" 1; cell_b "$REP" "$W" 2; cell_c "$REP" "$W"
  done
done
echo "[ao] DONE $(date +%H:%M:%S)" >> "$LOG"
touch ~/v/logs/anchor_only.done 2>/dev/null || true
