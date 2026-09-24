#!/usr/bin/env bash
# feat-184 (results/onset_prediction_served_opponent.md). One queue per card, jobs in order
# (caution (x)); each job writes output/logs/feat184_<job>.{done,fail}.
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
H2H="$PY analysis/order_averaged_h2h.py --deecho --out results"
run() {  # run <name> <cmd...>
  local name=$1; shift
  echo "[$name] start $(date '+%F %T')"
  if "$@"; then touch "output/logs/feat184_$name.done"; echo "[$name] exit=0 $(date '+%F %T')"
  else touch "output/logs/feat184_$name.fail"; echo "[$name] exit=$? $(date '+%F %T')"; fi
}
case "$1" in
  gpu2)
    export CUDA_VISIBLE_DEVICES=2
    run gen $PY h1.py --use-chat-template --k-values 10 --trajectories-per-prompt 1 --seeds 52 53 54 \
      --cap-neutral 200 --cap-factual 150 --cap-creative 150 --cap-val 0 --cap-test 0 \
      --cap-attack-train 0 --batch-size 48 --trust-remote-code --output-dir output/feat184/chat_k10
    [ -e output/logs/feat184_gen.done ] || exit 1
    run B1 $H2H --baseline-dir output/sweep_chat --metered-dir output/feat184/chat_k10 --k 10 \
      --anchor-dir output/sweep_chat --extra-dir output/sweep_chat --extra-token=-1 --extra-rank 1 \
      --extra-name opp_chat_draw2 --tag served_k10chat ;;
  gpu1)
    export CUDA_VISIBLE_DEVICES=1
    run A $H2H --tag deecho
    run B2 $H2H --baseline-dir output/sweep_chat --metered-dir output/phase2/conc_all --k 10 \
      --anchor-dir output/sweep_plain --extra-dir output/sweep_plain --extra-token=-1 \
      --extra-name opp_plain --tag served_committed ;;
  gpu4)
    export CUDA_VISIBLE_DEVICES=4
    run B3 $H2H --baseline-dir output/sweep_chat --metered-dir output/sweep_chat --k 1 \
      --anchor-dir output/sweep_chat --tag served_k1chat ;;
esac
