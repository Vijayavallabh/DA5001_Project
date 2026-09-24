#!/usr/bin/env bash
# feat-195 (results/onset_prediction_he_decoding.md) and feat-196 (results/onset_prediction_chat_grid.md).
# One queue per card, jobs in order (caution (x)); each job writes output/logs/feat195_<job>.{done,fail}.
# Judge queues wait on those sentinels as an OR over .done/.fail with a deadline (caution (c)).
#   bash scripts/run_feat195.sh gpu4 | gpu12 | chat | judge_he
# Cards default to the registration's local ones; on host B (2026-09-24 addenda) they are overridden:
#   GPU_MAIN=4 GPUS_70B=5,6 GPU_CHAT=7 GPU_J12=7 GPU_J34=5
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
L=output/logs
CAPS="--cap-neutral 200 --cap-factual 150 --cap-creative 150 --cap-val 0 --cap-test 0 --cap-attack-train 0"
HE="--temperature 0.7 --repetition-penalty 1.1"
ANCHOR=jacquelinehe/tinycomma-1.8b-llama3-tokenizer
H2H="$PY analysis/order_averaged_h2h.py --deecho --out results"
run() {  # run <name> <cmd...>
  local name=$1; shift
  echo "[$name] start $(date '+%F %T')"
  if "$@"; then touch "$L/feat195_$name.done"; echo "[$name] exit=0 $(date '+%F %T')"
  else local rc=$?; touch "$L/feat195_$name.fail"; echo "[$name] exit=$rc $(date '+%F %T')"; fi
}
wait_for() {  # wait_for <deadline_s> <job>... : every job has a .done or .fail; 0 iff all .done
  local deadline=$(( $(date +%s) + $1 )); shift
  for j in "$@"; do
    until [ -e "$L/feat195_$j.done" ] || [ -e "$L/feat195_$j.fail" ]; do
      [ "$(date +%s)" -ge "$deadline" ] && { echo "[wait] deadline passed waiting for $j"; return 2; }
      sleep 30
    done
    [ -e "$L/feat195_$j.done" ] || { echo "[wait] $j failed"; return 1; }
  done
}
case "$1" in
  gpu4)
    export CUDA_VISIBLE_DEVICES=${GPU_MAIN:-4}
    run t07_8b $PY h1.py --k-values -1 0 0.5 1 10 $HE --trajectories-per-prompt 1 $CAPS \
      --batch-size 48 --trust-remote-code --output-dir output/feat195/t07_8b
    run pool $PY h1.py --k-values 0 $HE --risky-model-path $ANCHOR --trajectories-per-prompt 64 $CAPS \
      --batch-size 256 --trust-remote-code --output-dir output/feat195/t07_pool64
    [ -e $L/feat195_pool.done ] && run rewards $PY analysis/pool_rewards.py \
      --sel-dir output/feat195/t07_pool64 --out results/selection_rewards64_t07.csv ;;
  gpu12)   # output/phase5/imit_llama70b's sharding: without it the 70B is pinned to one card (caution (q))
    export CUDA_VISIBLE_DEVICES=${GPUS_70B:-1,2}
    run t07_70b $PY h1.py --k-values -1 0.5 1 20 $HE --risky-model-path unsloth/Meta-Llama-3.1-70B \
      --parallelize --risky-device-map auto --max-memory 0=75GiB,1=70GiB \
      --trajectories-per-prompt 1 $CAPS --output-dir output/feat195/t07_70b ;;
  chat)   # feat-196: G1's same-host reference (feat-184 B3's command, committed text only), then the
          # new budgets, then J5 and J6, all on one card
    export CUDA_VISIBLE_DEVICES=${GPU_CHAT:-4}
    run G1ref $H2H --baseline-dir output/sweep_chat --metered-dir output/sweep_chat --k 1 \
      --anchor-dir output/sweep_chat --tag served_k1chat_hostB
    run chat $PY h1.py --use-chat-template --k-values 2 3 5 --trajectories-per-prompt 1 \
      --seeds 52 53 54 $CAPS --batch-size 48 --trust-remote-code --output-dir output/feat196/chat_grid
    [ -e $L/feat195_chat.done ] || exit 1
    run J5 $H2H --baseline-dir output/sweep_chat --anchor-dir output/sweep_chat \
      --metered-dir output/sweep_chat --k 0.5 --extra-dir output/feat196/chat_grid --extra-token 2 \
      --extra-name chat_k2 --tag chatgrid_k05
    run J6 $H2H --baseline-dir output/sweep_chat --anchor-dir output/sweep_chat \
      --metered-dir output/feat196/chat_grid --k 3 --extra-dir output/feat196/chat_grid --extra-token 5 \
      --extra-name chat_k5 --tag chatgrid_k3 ;;
  judge_he)     # after the pool is scored and the 70B is done; J1/J2 on GPU 1, J3/J4 on GPU 2
    wait_for 28800 t07_8b rewards t07_70b || exit 1
    S="--sel-dir output/feat195/t07_pool64 --rewards results/selection_rewards64_t07.csv"
    B="--baseline-dir output/feat195/t07_8b --anchor-dir output/feat195/t07_8b"
    ( export CUDA_VISIBLE_DEVICES=${GPU_J12:-1}
      run J1 $H2H $S $B --metered-dir output/feat195/t07_8b --k 10 --extra-dir output/feat195/t07_8b \
        --extra-token 0.5 --extra-name met8b_k0.5 --tag t07_8b_k10
      run J2 $H2H $S $B --metered-dir output/feat195/t07_8b --k 1 --extra-dir output/feat195/t07_70b \
        --extra-token=-1 --extra-name risky70b --tag t07_8b_k1 ) &
    ( export CUDA_VISIBLE_DEVICES=${GPU_J34:-2}
      run J3 $H2H $S $B --metered-dir output/feat195/t07_70b --k 20 --extra-dir output/feat195/t07_70b \
        --extra-token 0.5 --extra-name met70b_k0.5 --tag t07_70b_k20
      run J4 $H2H $S $B --metered-dir output/feat195/t07_70b --k 1 --tag t07_70b_k1 ) &
    wait ;;
  *) echo "usage: $0 gpu4|gpu12|chat|judge_he"; exit 2 ;;
esac
