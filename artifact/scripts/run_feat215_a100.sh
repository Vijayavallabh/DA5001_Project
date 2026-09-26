#!/usr/bin/env bash
# feat-215 addendum (results/onset_prediction_chained_fix.md): the eight arms first run on host B's H100s, re-run on
# the registered local A100s with their registered commands (run_feat215.sh's job lines, unchanged).
#   scripts/run_feat215_a100.sh comma      # GPU 4: comp_comma7b
#   scripts/run_feat215_a100.sh nm1984     # GPUs 1,2: the three 1984 arms, then output/chainfix/.nm1984_a100.done
#   scripts/run_feat215_a100.sh pathwise   # GPU 1, once the 1984 lane is done: comp8b_pathwise
#   scripts/run_feat215_a100.sh bankcap    # GPU 2, once the 1984 lane is done: the three bank caps
set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
O=output/chainfix
M=output/memorizing_llama8b
SNAP=hf_cache/models--unsloth--Meta-Llama-3.1-70B/snapshots/1b7306651142d0cc65d993076a250a6a82cf046c
C="--risky-model $SNAP --risky-device-map auto --max-memory 0=75GiB,1=70GiB --raw-prompt --seed-tokens 100 --windows 50 --batch-size 16"
KS="-1 0 1 1.5 3 5 10 20"
job () {  # name args...  (identical to run_feat215.sh)
  local name=$1; shift
  local d=$O/$name; mkdir -p $d
  .venv/bin/python analysis/composition_attack.py --modes chained "$@" \
    --out $d --figures $d --text-out $d/extracted.csv --queries-out $d/queries.jsonl > $d/run.log 2>&1
  echo "[feat215] $name exit $? $(date '+%F %T')"
}
after_nm () {  # the 1984 lane holds GPUs 1 and 2; wait for its sentinel, with a deadline
  local dl=$(( $(date +%s) + 7200 ))
  until [ -e $O/.nm1984_a100.done ]; do
    [ "$(date +%s)" -gt "$dl" ] && { echo "[feat215] deadline waiting for the 1984 lane"; exit 1; }
    sleep 20
  done
}
case "${1:-}" in
comma)    export CUDA_VISIBLE_DEVICES=4
          job comp_comma7b --safe-model common-pile/comma-v0.1-2t --risky-model output/phase4/memorizing_comma7b \
            --limit 100 --k-values -1 0 0.15 0.5 1 3 5 10 20 --windows 20 50 ;;
nm1984)   export CUDA_VISIBLE_DEVICES=1,2
          job nm/1984_greedy $C --split attack_train --novel 1984 --limit 8 --temperature 1.0 --repetition-penalty 1.0 --k-values -1 0 --greedy
          job nm/1984_B $C --split attack_train --novel 1984 --limit 8 --temperature 0.7 --repetition-penalty 1.1 --k-values $KS
          job nm/1984_A $C --split attack_train --novel 1984 --limit 8 --temperature 1.0 --repetition-penalty 1.0 --k-values $KS
          touch $O/.nm1984_a100.done ;;  # written whatever the exits were: each job's exit is in the log
pathwise) after_nm; export CUDA_VISIBLE_DEVICES=1
          job comp8b_pathwise --risky-model $M --limit 100 --k-values -1 0 1 3 5 10 20 50 --windows 20 50 --constraint pathwise ;;
bankcap)  after_nm; export CUDA_VISIBLE_DEVICES=2
          job bank_cap_k10_10 --risky-model $M --limit 100 --k-values 10 --windows 50 --bank-cap 10 --batch-size 16
          job bank_cap_k10_50 --risky-model $M --limit 100 --k-values 10 --windows 50 --bank-cap 50 --batch-size 16
          job bank_cap_k20_20 --risky-model $M --limit 100 --k-values 20 --windows 50 --bank-cap 20 --batch-size 16 ;;
*) echo "usage: $0 comma|nm1984|pathwise|bankcap"; exit 2 ;;
esac
echo "[feat215] lane $1 drained $(date '+%F %T')"
