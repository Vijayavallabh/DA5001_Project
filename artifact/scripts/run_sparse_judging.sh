#!/usr/bin/env bash
# feat-125/126/128: the judging passes, through the ONE endorsed instrument
# (analysis/order_averaged_h2h.py: every item in both orders, arms paired, one fixed opponent).
#
# Waits for the generation queue by looking for the string run_sparse_causal.sh writes when it
# drains -- a condition in the FILESYSTEM. Caution (c): never wait on `! pgrep -f X`, because the
# waiting shell's own command line contains X, so the negation can never become true.
#
# Every pass carries a --tag. The UNTAGGED file results/order_averaged_h2h.csv holds the paper's
# headline and must never be overwritten by an exploratory arm (feat-123 had to restore it).
#
# The first eight passes re-judge sel_n64 / sel_n1 / anchor_k0 unchanged, which is not waste: it
# yields EIGHT independent D1 estimates and so measures the cross-pass floor the reviewer asks us
# to quote, from a sample rather than from one duplicated pair.
#
# Usage: scripts/run_sparse_judging.sh [gpu]      (default 4; NEVER 3)
set -u
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh 2>/dev/null || true
GPU=${1:-4}
[ "$GPU" = "3" ] && { echo "GPU 3 is the 4GB T400; refusing"; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
GEN_LOG=output/logs/sparse_causal.log
LOG=output/logs/sparse_judging.log
mkdir -p output/logs

echo "[judge] $(date +%H:%M:%S) waiting for the generation queue to drain" >> $LOG
until grep -q "QUEUE DRAINED" "$GEN_LOG" 2>/dev/null; do sleep 60; done
echo "[judge] $(date +%H:%M:%S) generation done, judging starts" >> $LOG

CANON=results/order_averaged_h2h.csv
BEFORE=$(md5sum $CANON | cut -d' ' -f1)

judge () {  # judge <tag> <extra args...>
  local tag="$1"; shift
  echo "[judge] $(date +%H:%M:%S) START $tag" >> $LOG
  .venv/bin/python analysis/order_averaged_h2h.py --out results --tag "$tag" "$@" \
    >> output/logs/h2h_$tag.log 2>&1
  echo "[judge] $(date +%H:%M:%S) END   $tag rc=$?" >> $LOG
}

# --- feat-125: the sparse causal policy, budget curve then placement search -----------------------
for ARM in b2.08_t0 b4.16_t0 b64_t0 b4.16_t1 b4.16_t2 b4.16_t4 b4.16_t8; do
  D=output/phase5/sparse_$ARM
  [ -d "$D" ] && judge "sparse_$ARM" --metered-dir "$D" --k 1e-9 --metered-constraint kl
done

# --- feat-126: alpha=8 at k=3, against the anchor alone through the same instrument ---------------
judge renyi8 --metered-dir output/phase5/renyi8_k3_full --k 3.0 --metered-constraint "renyi:8"

# --- feat-128: the bigger safe model. Its rank-0 draw IS Comma-7B served alone, so this pass puts
#     Comma-7B-alone and TinyComma-alone in ONE pass against ONE opponent (no new generation).
judge comma7b_alone --sel-dir output/phase5/sel_comma7b_64 \
  --rewards results/selection_rewards64_comma7b.csv --n 64

AFTER=$(md5sum $CANON | cut -d' ' -f1)
if [ "$BEFORE" = "$AFTER" ]; then
  echo "[judge] $(date +%H:%M:%S) canonical headline INTACT ($AFTER)" >> $LOG
else
  echo "[judge] $(date +%H:%M:%S) *** CANONICAL FILE CHANGED -- $BEFORE -> $AFTER ***" >> $LOG
fi
echo "[judge] $(date +%H:%M:%S) JUDGING DRAINED" >> $LOG
