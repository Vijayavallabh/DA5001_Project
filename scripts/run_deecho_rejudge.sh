#!/usr/bin/env bash
# results/deecho_rejudge_note.md: the older judged passes re-run on de-echoed text (caution (bc)).
# Each job is its producing command on record plus --deecho and a tag ending _deecho, so no
# committed CSV is touched (caution (ax)). One queue shell per card set, jobs in order (caution (x)).
# Usage: run_deecho_rejudge.sh <q56|q7>     (host B; writes ~/v/logs/dr_<job>.{done,fail})
set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
H2H=".venv/bin/python analysis/order_averaged_h2h.py --deecho --seed 7717 --n 64 --out results"
PB="microsoft/Phi-3.5-mini-instruct"; PC="meta-llama/Meta-Llama-3.1-8B-Instruct"
HEAD="--sel-dir output/xfer/sel_anchor64 --metered-dir output/xfer/conc_all --anchor-dir output/sweep_plain --rewards results/selection_rewards64.csv --k 10"
run() {  # run <job> <args...>
  local j=$1; shift
  rm -f ~/v/logs/dr_$j.done ~/v/logs/dr_$j.fail
  echo "[dr:$j] START $(date '+%F %T')"
  if $H2H "$@" > "output/logs/dr_$j.log" 2>&1; then touch ~/v/logs/dr_$j.done; R=0; else touch ~/v/logs/dr_$j.fail; R=1; fi
  echo "[dr:$j] rc=$R $(date '+%F %T')"
}
wl() {  # wl <job> <corpus> <datadir> <cell> <k> <tag>
  local S=output/$2/sel_anchor256; [ -d "$S" ] || S=output/$2/sel_anchor64
  run "$1" --judge $PB --sel-dir "$S" --metered-dir "output/$2/$4" --anchor-dir "$S" \
    --baseline-dir "output/$2/baseline" --rewards "results/$2_rewards.csv" --data-dir "$3" --k "$5" --tag "$6"
}
mkdir -p output/logs
case "${1:?q56|q7}" in
  q56)
    export CUDA_VISIBLE_DEVICES=5,6
    run qwen72b --judge Qwen/Qwen2.5-72B-Instruct --device-map auto $HEAD --baseline-dir output/sweep_plain --tag _qwen72b_deecho
    run mixtral --judge mistralai/Mixtral-8x7B-Instruct-v0.1 --device-map auto $HEAD --baseline-dir output/sweep_plain --tag _mixtral_deecho ;;
  q7)
    export CUDA_VISIBLE_DEVICES=7
    .venv/bin/python analysis/deecho_rejudge.py --coverage --out results > output/logs/dr_coverage.log 2>&1 \
      && touch ~/v/logs/dr_coverage.done || touch ~/v/logs/dr_coverage.fail
    for o in qwen05b qwen15b qwen3b; do
      run opp_${o}_B --judge $PB $HEAD --baseline-dir output/opponent_$o --tag _opp_${o}_deecho
    done
    run opp2_B --judge $PB $HEAD --baseline-dir output/opponent_qwen14b --tag _opp2_deecho
    wl wl_mtb mtb data/bench/mtbench conc_bind 1.0 _mtb_conc_bind_deecho
    wl wl_gutenberg gutenberg data/bench/gutenberg conc_bind 0.9 _gutenberg_conc_bind_deecho
    wl wl_unseenbooks unseenbooks data/bench/unseenbooks conc_bind 1.0 _unseenbooks_conc_bind_deecho
    wl wl_cotaeval cotaeval_qa data/bench/cotaeval_qa conc_bind 1.4 _cotaeval_qa_conc_bind_deecho
    run wl_alpaca --judge $PB --sel-dir output/mixpow/sel_anchor64 --metered-dir output/mixpow/conc_k10 \
      --anchor-dir output/mixpow/sel_anchor64 --baseline-dir output/mixpow/baseline \
      --rewards results/mixpow_rewards64.csv --data-dir data/bench/alpaca --k 1 --tag _mixpowk_judgeB_deecho
    run wl_wscope_c --judge $PB --sel-dir output/wscope/a_sel --metered-dir output/wscope/c_conc \
      --anchor-dir output/wscope/a_sel --baseline-dir output/wscope/a_baseline \
      --rewards results/wscope_rewards64_a.csv --data-dir data --k 0.9 --tag _wscope_c_deecho
    run opp_committed_C --judge $PC $HEAD --baseline-dir output/sweep_plain --tag _opp_committed_judgeC_deecho
    for o in qwen05b qwen15b qwen3b; do
      run opp_${o}_C --judge $PC $HEAD --baseline-dir output/opponent_$o --tag _opp_${o}_judgeC_deecho
    done
    run qwen14b --judge Qwen/Qwen2.5-14B-Instruct --device-map auto $HEAD --baseline-dir output/sweep_plain --tag _qwen14b_deecho
    run gemma27b --judge google/gemma-2-27b-it --device-map auto $HEAD --baseline-dir output/sweep_plain --tag _gemma27b_deecho ;;
esac
echo "[dr:$1] queue finished $(date '+%F %T')"
touch ~/v/logs/dr_queue_$1.finished
