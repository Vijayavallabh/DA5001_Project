#!/usr/bin/env bash
# feat-209 (results/onset_prediction_nonempty_tables.md): eight rows of Table 2 re-judged with the non-empty
# rule's pick beside the committed pick, judge B, host B. One queue, jobs in order (caution (x)); it starts
# once feat-208's GPU-4 queue has finished (an OR over sentinels with a deadline, caution (c)).
#   GPU=4 bash scripts/run_feat209.sh
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache" CUDA_VISIBLE_DEVICES=${GPU:-4}
PY=.venv/bin/python
L=output/logs
N=output/feat209
run() {  # run <name> <cmd...>
  local name=$1; shift
  echo "[$name] start $(date '+%F %T')"
  if "$@"; then touch "$L/feat209_$name.done"; echo "[$name] exit=0 $(date '+%F %T')"
  else local rc=$?; touch "$L/feat209_$name.fail"; echo "[$name] exit=$rc $(date '+%F %T')"; fi
}
deadline=$(( $(date +%s) + 21600 ))
until [ -e $L/feat208_judgeG_blk10n64s2.done ] || [ -e $L/feat208_judgeG_blk10n64s2.fail ] \
      || [ -e $L/feat208_blk10n64s2.fail ]; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[feat209] deadline passed"; exit 2; }
  sleep 30
done
run arms $PY analysis/nonempty_arms.py --out $N
[ -e $L/feat209_arms.done ] || exit 1
h2h() {  # h2h <row> <pool> [args...]
  local row=$1 pool=$2; shift 2
  run "$row" $PY analysis/order_averaged_h2h.py --deecho --seed 7717 --out results --extra-dir $N/$pool \
    --extra-token nonempty --extra-name sel_nonempty --tag "nonempty_$row" "$@"
}
T07="--sel-dir output/feat195/t07_pool64 --rewards results/selection_rewards64_t07.csv --baseline-dir output/feat195/t07_8b --anchor-dir output/feat195/t07_8b"
h2h headline pool
h2h he70b_k20 pool --metered-dir output/phase5/imit_llama70b --k 20 --anchor-dir output/phase5/imit_llama70b
h2h he70b_k05 pool --metered-dir output/phase5/imit_llama70b --k 0.5 --anchor-dir output/phase5/imit_llama70b
h2h chat_k10 pool --baseline-dir output/sweep_chat --metered-dir output/feat184/chat_k10 --k 10 --anchor-dir output/sweep_chat
h2h chat_k3 pool --baseline-dir output/sweep_chat --metered-dir output/feat196/chat_grid --k 3 --anchor-dir output/sweep_chat
h2h t07_8b_k10 t07 $T07 --metered-dir output/feat195/t07_8b --k 10
h2h t07_70b_k20 t07 $T07 --metered-dir output/feat195/t07_70b --k 20
h2h ab70_k01 comma7b --n 64 --sel-dir output/phase5/sel_comma7b_64 --rewards results/selection_rewards64_comma7b.csv \
  --metered-dir output/anchoredbyte/comma7b_70b --k 0.1 --anchor-dir output/phase5/sel_comma7b_64 --baseline-dir output/sweep_plain
echo "[feat209] finished $(date '+%F %T')"
