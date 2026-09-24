#!/usr/bin/env bash
# feat-185 (results/onset_prediction_he_config.md): judge-only, on GPU 4 after feat-184's run B3.
# Waits on B3's sentinel -- an OR over .done/.fail (caution (c), ninth incident) -- with a deadline.
set -u
cd "$(dirname "$0")/.." || exit 1
LOGS=${FEAT185_LOGS:-output/logs}
deadline=$(( $(date +%s) + ${FEAT185_WAIT_S:-14400} ))
until [ -e "$LOGS/feat184_B3.done" ] || [ -e "$LOGS/feat184_B3.fail" ]; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[feat185] gave up waiting for B3"; exit 2; }
  sleep "${FEAT185_POLL_S:-30}"
done
echo "[feat185] B3 finished; starting $(date '+%F %T')"
[ -n "${FEAT185_DRY:-}" ] && exit 0
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache" CUDA_VISIBLE_DEVICES=4
H2H=".venv/bin/python analysis/order_averaged_h2h.py --deecho --out results"
D=output/phase5/imit_llama70b
run() { local n=$1; shift; echo "[$n] start $(date '+%F %T')"
  if "$@"; then touch "$LOGS/feat185_$n.done"; echo "[$n] exit=0 $(date '+%F %T')"
  else touch "$LOGS/feat185_$n.fail"; echo "[$n] failed $(date '+%F %T')"; fi; }
run Ca $H2H --metered-dir $D --k 20 --anchor-dir $D --extra-dir $D --extra-token 1 --extra-name met70b_k1 --tag he70b_k20
run Cb $H2H --metered-dir $D --k 0.5 --anchor-dir $D --extra-dir $D --extra-token=-1 --extra-name risky70b --tag he70b_k05
