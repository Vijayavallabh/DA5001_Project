#!/usr/bin/env bash
# feat-201 (results/onset_prediction_blockwise.md): selection spent in installments. One queue shell per
# card, jobs in order (caution (x)); each job writes output/logs/feat201_<job>.{done,fail}, and a job
# whose generation failed is not judged.
#   GPU=4 bash scripts/run_feat201.sh A     # blk10n64
#   GPU=5 bash scripts/run_feat201.sh B     # the gate arms, blk200n64, blk25n64
#   GPU=6 bash scripts/run_feat201.sh C     # blk50n64, the ablation, the matched-certificate arms, leakage
#   GPU=7 bash scripts/run_feat201.sh D     # the planner
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache" CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
PY=.venv/bin/python
L=output/logs
O=output/feat201
mkdir -p $L $O
run() {  # run <name> <cmd...>
  local name=$1; shift
  echo "[$name] start $(date '+%F %T')"
  if "$@"; then touch "$L/feat201_$name.done"; echo "[$name] exit=0 $(date '+%F %T')"
  else local rc=$?; touch "$L/feat201_$name.fail"; echo "[$name] exit=$rc $(date '+%F %T')"; fi
}
gen() {  # gen <arm> <args...>
  local arm=$1; shift
  run "$arm" $PY analysis/blockwise_selection.py --out-dir $O/$arm --tag $arm "$@"
}
judge() {  # judge <arm>, only if its generation finished
  [ -e $L/feat201_$1.done ] || return 0
  run "judge_$1" $PY analysis/order_averaged_h2h.py --deecho --extra-dir $O/$1 --extra-token $1 \
    --extra-name $1 --tag blockwise_$1 --out results
}
case "${1:?queue A|B|C|D}" in
  A) gen blk10n64 --block-len 10 --n 64; judge blk10n64 ;;
  B) gen blk200n1 --block-len 200 --n 1
     gen blk200n1_shipped --block-len 200 --n 1 --shipped-sampling --smoke 100
     [ -e $L/feat201_blk200n1.done ] && [ -e $L/feat201_blk200n1_shipped.done ] && \
       run gate $PY analysis/blockwise_gate.py --out results
     judge blk200n1
     gen blk200n64 --block-len 200 --n 64; judge blk200n64
     gen blk25n64 --block-len 25 --n 64; judge blk25n64 ;;
  C) gen blk50n64 --block-len 50 --n 64; judge blk50n64
     gen blk50n64_reward --block-len 50 --n 64 --scorer reward; judge blk50n64_reward
     gen blk100n8 --block-len 100 --n 8; judge blk100n8
     gen blk67n4 --block-len 67 --n 4; judge blk67n4
     gen blk34n2 --block-len 34 --n 2; judge blk34n2
     gen blk10n64_memoriser --block-len 10 --n 64 --scorer memoriser --corpus passages --limit 100 \
       --risky-model output/memorizing_llama8b --results results ;;
  D) gen blk10n64_planner --block-len 10 --n 64 --scorer planner; judge blk10n64_planner ;;
esac
echo "[queue $1] finished $(date '+%F %T')"
