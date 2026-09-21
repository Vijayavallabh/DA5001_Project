#!/usr/bin/env bash
# results/onset_prediction_batch_width.md (feat-163): does batching move the PRICE RATIO, or only
# the bill? One exclusively-held card, box otherwise idle, every cell twice with the second rep in
# reverse width order so drift cancels (caution (ae)).
#
# --cap-neutral W with --batch-size W is exactly one batch per seed group, so the width is W by
# construction; the loader is removed per width by running one AND two completions, which is the
# same construction feat-162 used for the metered path and not a fit.
#
# Usage: run_batch_width.sh <gpu>
set -u
GPU=${1:-0}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
LOG=output/logs/batch_width.log
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"

stamp() { python3 -c 'import time;print(f"{time.time():.3f}")'; }
secs()  { python3 -c "print(f'{$2-$1:.3f}')"; }

cell() {   # rep path W tpp   (path is ANCHOR or MET)
  local REP=$1 P=$2 W=$3 TPP=$4 K D T0 T1
  [ "$P" = "ANCHOR" ] && K=0.0 || K=10.0
  D=output/phase5/bw_${P}_w${W}_t${TPP}_r${REP}
  rm -rf "$D"; T0=$(stamp)
  env $E .venv/bin/python h1.py --k-values "$K" --trajectories-per-prompt "$TPP" \
    --cap-neutral "$W" --cap-creative 0 --cap-factual 0 \
    --cap-val 0 --cap-test 0 --cap-attack-train 0 \
    --max-new-tokens 200 --batch-size "$W" --output-dir "$D" >> "$LOG" 2>&1
  T1=$(stamp)
  echo "[width] $P rep=$REP W=$W tpp=$TPP seconds=$(secs "$T0" "$T1") dir=$D" >> "$LOG"
}

echo "[width] START $(date +%H:%M:%S) gpu=$GPU" >> "$LOG"
echo "[width] box at start: $(env -u LD_LIBRARY_PATH nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')" >> "$LOG"

for W in 8 16 32 64 128 200; do
  for P in ANCHOR MET; do cell 1 "$P" "$W" 1; cell 1 "$P" "$W" 2; done
done
for W in 200 128 64 32 16 8; do
  for P in MET ANCHOR; do cell 2 "$P" "$W" 2; cell 2 "$P" "$W" 1; done
done

echo "[width] box at end: $(env -u LD_LIBRARY_PATH nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')" >> "$LOG"
echo "[width] DONE $(date +%H:%M:%S)" >> "$LOG"
touch ~/v/logs/batch_width.done 2>/dev/null || true
