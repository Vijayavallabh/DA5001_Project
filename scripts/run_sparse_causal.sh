#!/usr/bin/env bash
# feat-125/126: the two arms a second reviewer's strongest points demand.
#
#   feat-125  the SPARSE CAUSAL POLICY Proposition 3 permits and the paper never built.
#             --initial-bank B with a negligible refill = a sequence budget constant in T;
#             --spend-threshold tau decides WHERE it goes, causally. Grid: B at tau=0 (the budget
#             curve) and tau at B=log 64 (the placement search, tilted to give the causal policy
#             its best shot). Bands: results/onset_prediction_sparse_causal.md
#   feat-126  is alpha=8's 80x extraction cut a repair, or the trivial horn? The 500-prompt
#             regeneration the judged comparison needs. Bands: results/onset_prediction_alpha_trivial.md
#
# ONE queue shell, ONE card, jobs in series -- caution (x): no PID is ever captured, the sequencing
# is this shell's own. Caution (ab): gpu_env.sh strips the shadowing driver dir so NVML works.
# COMMON is sweep_plain's own argument list, because that run is the k=0 control every arm is
# judged against and the batch size is part of the seed (caution (u)).
#
# Usage: scripts/run_sparse_causal.sh [gpu]      (default 4; NEVER 3)
set -u
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh 2>/dev/null || true
GPU=${1:-4}
[ "$GPU" = "3" ] && { echo "GPU 3 is the 4GB T400; refusing"; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache

COMMON="--trajectories-per-prompt 3 --cap-neutral 200 --cap-val 150 --cap-test 150 \
--cap-attack-train 100 --cap-factual 150 --cap-creative 150 --batch-size 48 --trust-remote-code \
--skip-existing"
LOG=output/logs/sparse_causal.log
mkdir -p output/logs output/phase5

run () {  # run <outdir> <extra args...>
  local dir="$1"; shift
  mkdir -p "$dir"
  echo "[sparse] $(date +%H:%M:%S) START $dir :: $*" >> $LOG
  .venv/bin/python h1.py $COMMON "$@" --output-dir "$dir" >> "$dir/run.log" 2>&1
  echo "[sparse] $(date +%H:%M:%S) END   $dir rc=$?" >> $LOG
}

# --- feat-125, the budget curve at greedy placement (tau unset = spend as soon as it binds) ------
run output/phase5/sparse_b2.08_t0 --k-values 1e-9 --initial-bank 2.0794 --no-prefix-debt
run output/phase5/sparse_b4.16_t0 --k-values 1e-9 --initial-bank 4.1589 --no-prefix-debt
run output/phase5/sparse_b64_t0   --k-values 1e-9 --initial-bank 64.0   --no-prefix-debt

# --- feat-125, the placement search at B = log 64, the budget selection spends at n=64 -----------
for TAU in 1 2 4 8; do
  run "output/phase5/sparse_b4.16_t$TAU" --k-values 1e-9 --initial-bank 4.1589 \
      --spend-threshold "$TAU" --no-prefix-debt
done

# --- feat-126, alpha=8 at k=3 on the FULL 500-prompt ordinary set (the arm on record is 150) -----
run output/phase5/renyi8_k3_full --k-values 3.0 --constraint renyi:8

echo "[sparse] $(date +%H:%M:%S) QUEUE DRAINED" >> $LOG
