#!/usr/bin/env bash
# feat-202..206 (results/onset_prediction_{factscore_oracle,vetting_t07,full_response,anchoredbyte_t07,
# chat_onset}.md), queued on host B behind feat-201's queues. One queue shell per card, jobs in order
# (caution (x)); each job writes output/logs/feat20x_<job>.{done,fail}. A queue starts when a sentinel
# of the feat-201 queue it follows exists -- an OR over files, with a deadline (caution (c)).
#   bash scripts/run_feat202_206.sh q7 | q5 | q6 | q456
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
L=output/logs
TC=jacquelinehe/tinycomma-1.8b-llama3-tokenizer
C7=common-pile/comma-v0.1-2t
run() {  # run <name> <cmd...>
  local name=$1; shift
  echo "[$name] start $(date '+%F %T')"
  if "$@"; then touch "$L/$name.done"; echo "[$name] exit=0 $(date '+%F %T')"
  else local rc=$?; touch "$L/$name.fail"; echo "[$name] exit=$rc $(date '+%F %T')"; fi
}
wait_any() {  # wait_any <deadline_s> <file>... : returns 0 once ANY file exists, 2 at the deadline
  local deadline=$(( $(date +%s) + $1 )); shift
  while :; do
    for f in "$@"; do [ -e "$f" ] && return 0; done
    [ "$(date +%s)" -ge "$deadline" ] && { echo "[wait] deadline passed: $*"; return 2; }
    sleep 30
  done
}
vet() {  # vet <tag> <model> <L>: the screen at temperature 0.7 (feat-203)
  run "feat203_vet_$1_L$3" $PY analysis/selection_extraction.py --safe-model "$2" \
    --risky-model output/memorizing_llama8b --raw-prompt --split test --novel harry_potter --limit 50 \
    --seed-tokens "$3" --max-new-tokens 200 --n-values 1 8 64 --batch-size 8 --temperature 0.7 \
    --prefix "vett07_L$3_$1" --out results
}
case "${1:?queue}" in
  q7)  # after feat-201 queue D (the planner): feat-202
    export CUDA_VISIBLE_DEVICES=${GPU:-7}
    wait_any 21600 $L/feat201_judge_blk10n64_planner.done $L/feat201_judge_blk10n64_planner.fail \
      $L/feat201_blk10n64_planner.fail || exit 1
    run feat202_dump $PY analysis/factscore_oracle.py --dump
    [ -e $L/feat202_dump.done ] || exit 1
    run feat202_factscore $PY analysis/he_metrics.py --factscore --dump output/he_metrics/pool64_factual.jsonl \
      --tag _pool64 --out results
    run feat202_factuality $PY analysis/factscore_oracle.py --factuality
    [ -e $L/feat202_factscore.done ] && [ -e $L/feat202_factuality.done ] && \
      run feat202_report $PY analysis/factscore_oracle.py --report ;;
  q5)  # after feat-201 queue B: feat-204, feat-206, feat-203's TinyComma rungs and both S measurements
    export CUDA_VISIBLE_DEVICES=${GPU:-5}
    wait_any 21600 $L/feat201_judge_blk25n64.done $L/feat201_judge_blk25n64.fail $L/feat201_blk25n64.fail || exit 1
    run feat204_judge $PY analysis/order_averaged_h2h.py --deecho --judge-max-chars 0 --tag fullresp --out results
    run feat206_onset $PY analysis/composition_attack.py --safe-model $TC --risky-model output/memorizing_llama8b \
      --use-chat-template --k-values -1 0 1.5 2 2.6 3.2 3.8 4.5 --modes single --windows 50 --limit 100 \
      --out output/feat206/chat_onset --text-out output/feat206/chat_onset/composition_extracted.csv \
      --queries-out output/feat206/chat_onset/queries.jsonl
    [ -e $L/feat206_onset.done ] && run feat206_ci $PY analysis/onset_ci.py \
      --comp output/feat206/chat_onset/composition.csv --s-x 3.239 --label "TinyComma-1.8B + mem. Llama-3.1-8B, chat" \
      --thresh 0.01 --out output/feat206/chat_onset
    for Lr in 20 100 200; do vet tinycomma $TC $Lr; done
    run feat203_regimes_tc $PY analysis/regimes.py --model $TC --temperature 0.7 \
      --out results/regimes_copybench_t07nopen_tinycomma.csv
    run feat203_regimes_c7 $PY analysis/regimes.py --model $C7 --temperature 0.7 \
      --out results/regimes_copybench_t07nopen_comma7b.csv
    run feat205_regimes_c7 $PY analysis/regimes.py --model $C7 --temperature 0.7 --repetition-penalty 1.1 \
      --out results/regimes_copybench_t07_comma7b.csv
    run feat203_tqa $PY analysis/tqa_vacuity.py --temperature 0.7 --out results/tqa_vacuity_t07.csv ;;
  q6)  # after feat-201 queue C: feat-203's Comma-7B rungs
    export CUDA_VISIBLE_DEVICES=${GPU:-6}
    wait_any 21600 $L/feat201_blk10n64_memoriser.done $L/feat201_blk10n64_memoriser.fail || exit 1
    for Lr in 20 100 200; do vet comma7b $C7 $Lr; done ;;
  q456)  # after feat-201 queue A and the two queues above: feat-205 on three cards, then the 70B control
    wait_any 43200 $L/feat201_judge_blk10n64.done $L/feat201_judge_blk10n64.fail $L/feat201_blk10n64.fail || exit 1
    wait_any 43200 $L/feat203_tqa.done $L/feat203_tqa.fail || exit 1
    wait_any 43200 $L/feat203_vet_comma7b_L200.done $L/feat203_vet_comma7b_L200.fail || exit 1
    for K in 0.1 0; do
      run feat205_ab07_k$K env CUDA_VISIBLE_DEVICES=${GPUS3:-4,5,6} $PY analysis/anchoredbyte_decode.py \
        --risky unsloth/Meta-Llama-3.1-70B --k $K --batch-size 32 --temperature 0.7 --repetition-penalty 1.1 \
        --out-dir output/feat205/ab07
    done
    [ -e $L/feat205_ab07_k0.1.done ] && [ -e $L/feat205_ab07_k0.done ] && \
      run feat205_judge env CUDA_VISIBLE_DEVICES=${GPU1:-4} $PY analysis/order_averaged_h2h.py --deecho \
        --baseline-dir output/feat195/t07_8b --sel-dir output/feat195/t07_pool64 \
        --rewards results/selection_rewards64_t07.csv --metered-dir output/feat205/ab07 --k 0.1 \
        --anchor-dir output/feat205/ab07 --tag ab07_k01 --out results
    run feat203_vet_llama70b_L100 env CUDA_VISIBLE_DEVICES=${GPUS2:-5,6} $PY analysis/selection_extraction.py \
      --safe-model $TC --risky-model unsloth/Meta-Llama-3.1-70B --risky-device-map auto \
      --max-memory 0=75GiB,1=70GiB --raw-prompt --split test --novel harry_potter --limit 50 --seed-tokens 100 \
      --max-new-tokens 200 --n-values 1 8 64 --batch-size 8 --temperature 0.7 --prefix vett07_L100_llama70b \
      --out results ;;
esac
echo "[queue $1] finished $(date '+%F %T')"
