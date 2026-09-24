#!/usr/bin/env bash
# feat-193 (results/onset_prediction_cotaeval_infringement.md): CoTaEval's infringement split with the
# benchmark's own per-item prefixes. One queue shell on one card (caution (x)); starts when the card's
# previous queue has written its last marker (OR over .done/.fail, with a deadline).
# Usage: run_cotaeval_inf.sh <gpu> [<marker to wait for, without .done/.fail>]
set -u
GPU=${1:?gpu}; AFTER=${2:-}
cd "$(dirname "$0")/.." || exit 1
M=${CTA_MARKS:-output/logs/marks}; mkdir -p "$M" output/logs output/cotaeval_inf
if [ -n "$AFTER" ]; then
  dl=$(( $(date +%s) + 28800 ))
  until [ -e "$M/$AFTER.done" ] || [ -e "$M/$AFTER.fail" ]; do
    [ "$(date +%s)" -ge "$dl" ] && { echo "gave up waiting for $AFTER"; exit 2; }; sleep 60; done
fi
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
A=jacquelinehe/tinycomma-1.8b-llama3-tokenizer
C="--data-dir data/bench/cotaeval_inf --cap-factual 500 --cap-neutral 0 --cap-creative 0 --cap-val 0 \
   --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 --trust-remote-code"
job() { local n=$1; shift; rm -f $M/cta_$n.done $M/cta_$n.fail
  echo "[cta:$n] start $(date '+%F %T') gpu=$GPU" >> output/logs/cta_$n.log
  if "$@" >> output/logs/cta_$n.log 2>&1; then touch $M/cta_$n.done; else touch $M/cta_$n.fail; fi
  echo "[cta:$n] end $(date '+%F %T')" >> output/logs/cta_$n.log; }
job arms .venv/bin/python h1.py --k-values -1 0 0.5 1 10 --trajectories-per-prompt 1 --batch-size 48 $C \
  --output-dir output/cotaeval_inf/arms
job pool .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 64 --batch-size 64 $C \
  --safe-model-path $A --risky-model-path $A --output-dir output/cotaeval_inf/pool
job reward .venv/bin/python analysis/selection_scaling.py --gen-dir output/cotaeval_inf/pool \
  --baseline-dir output/cotaeval_inf/arms --reward-cache results/selection_rewards64_cotaeval_inf.csv \
  --rewards-only --batch-size 8 --out results
job score .venv/bin/python analysis/cotaeval_infringement.py --out results
