#!/usr/bin/env bash
# feat-208 (results/onset_prediction_blockwise_replication.md): feat-201's installment comparison on a
# disjoint draw, read by judge B and judge G; then feat-205 (AnchoredByte at 0.7/1.1) and feat-203's 70B
# positive control, re-queued here after the first q456 queue was stopped before it started anything.
# One queue shell per card or card set, jobs in order (caution (x)); waits are ORs over sentinels with a
# deadline (caution (c)).
#   bash scripts/run_feat208.sh a | b | c | ab
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
L=output/logs
O=output/feat208
G=google/gemma-2-27b-it
TC=jacquelinehe/tinycomma-1.8b-llama3-tokenizer
mkdir -p $L $O
run() {  # run <name> <cmd...>
  local name=$1; shift
  echo "[$name] start $(date '+%F %T')"
  if "$@"; then touch "$L/$name.done"; echo "[$name] exit=0 $(date '+%F %T')"
  else local rc=$?; touch "$L/$name.fail"; echo "[$name] exit=$rc $(date '+%F %T')"; fi
}
wait_any() {  # wait_any <deadline_s> <file>... : 0 once ANY file exists, 2 at the deadline
  local deadline=$(( $(date +%s) + $1 )); shift
  while :; do
    for f in "$@"; do [ -e "$f" ] && return 0; done
    [ "$(date +%s)" -ge "$deadline" ] && { echo "[wait] deadline passed: $*"; return 2; }
    sleep 30
  done
}
h2h() {  # h2h <job> <dir> <token> <tag> [judge args...]
  local job=$1 dir=$2 tok=$3 tag=$4; shift 4
  run "$job" $PY analysis/order_averaged_h2h.py --deecho --extra-dir "$dir" --extra-token "$tok" \
    --extra-name "$tok" --tag "$tag" --out results "$@"
}
arm() {  # arm <L> : draw, then judge B and judge G
  local a=blk$1n64s2
  run feat208_$a $PY analysis/blockwise_selection.py --out-dir $O/$a --tag $a --block-len $1 --n 64 --seed 20260926
  [ -e $L/feat208_$a.done ] || return 0
  h2h feat208_judgeB_$a $O/$a $a blockwise_$a
  h2h feat208_judgeG_$a $O/$a $a blockwise_${a}_judgeG --judge $G --device-map auto
}
case "${1:?queue}" in
  a)  # GPU 4: judge G, post hoc, on feat-201's own texts; then the L=10 re-draw
    export CUDA_VISIBLE_DEVICES=${GPU:-4}
    for x in blk10n64 blk25n64 blk50n64 blk200n64; do
      h2h feat208_posthocG_$x output/feat201/$x $x blockwise_${x}_judgeG --judge $G --device-map auto
    done
    arm 10 ;;
  b)  # GPU 5, after feat-203's q5 queue: the L=200 re-draw
    export CUDA_VISIBLE_DEVICES=${GPU:-5}
    wait_any 14400 $L/feat203_tqa.done $L/feat203_tqa.fail || exit 1
    arm 200 ;;
  c)  # GPU 6, after feat-203's q6 queue: the L=25 re-draw
    export CUDA_VISIBLE_DEVICES=${GPU:-6}
    wait_any 14400 $L/feat203_vet_comma7b_L200.done $L/feat203_vet_comma7b_L200.fail || exit 1
    arm 25 ;;
  ab)  # GPUs 5,6,7 once b, c and feat-202 are finished: feat-205, then the 70B control of feat-203
    wait_any 28800 $L/feat208_judgeG_blk200n64s2.done $L/feat208_judgeG_blk200n64s2.fail \
      $L/feat208_blk200n64s2.fail || exit 1
    wait_any 28800 $L/feat208_judgeG_blk25n64s2.done $L/feat208_judgeG_blk25n64s2.fail \
      $L/feat208_blk25n64s2.fail || exit 1
    wait_any 28800 $L/feat202_report.done $L/feat202_report.fail $L/feat202_factscore.fail \
      $L/feat202_factuality.fail || exit 1
    for K in 0.1 0; do
      run feat205_ab07_k$K env CUDA_VISIBLE_DEVICES=${GPUS3:-5,6,7} $PY analysis/anchoredbyte_decode.py \
        --risky unsloth/Meta-Llama-3.1-70B --k $K --batch-size 32 --temperature 0.7 --repetition-penalty 1.1 \
        --out-dir output/feat205/ab07
    done
    [ -e $L/feat205_ab07_k0.1.done ] && [ -e $L/feat205_ab07_k0.done ] && \
      run feat205_judge env CUDA_VISIBLE_DEVICES=${GPU1:-5} $PY analysis/order_averaged_h2h.py --deecho \
        --baseline-dir output/feat195/t07_8b --sel-dir output/feat195/t07_pool64 \
        --rewards results/selection_rewards64_t07.csv --metered-dir output/feat205/ab07 --k 0.1 \
        --anchor-dir output/feat205/ab07 --tag ab07_k01 --out results
    run feat203_vet_llama70b_L100 env CUDA_VISIBLE_DEVICES=${GPUS2:-6,7} $PY analysis/selection_extraction.py \
      --safe-model $TC --risky-model unsloth/Meta-Llama-3.1-70B --risky-device-map auto \
      --max-memory 0=75GiB,1=70GiB --raw-prompt --split test --novel harry_potter --limit 50 --seed-tokens 100 \
      --max-new-tokens 200 --n-values 1 8 64 --batch-size 8 --temperature 0.7 --prefix vett07_L100_llama70b \
      --out results ;;
esac
echo "[queue $1] finished $(date '+%F %T')"
