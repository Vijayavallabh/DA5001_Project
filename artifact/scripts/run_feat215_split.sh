#!/usr/bin/env bash
# feat-215, split across two hosts at the user's instruction (2026-09-26 01:36: "use host B's free GPUs to speed it
# up"). Every job is scripts/run_feat215.sh's command, unchanged; only where it runs moved. The two local queue shells
# were stopped by PID with their in-flight arms (phase1, nm/hp1_B) left running; the local lanes wait for those PIDs.
#   local:  scripts/run_feat215_split.sh kl <phase1 python PID>   # GPU 4: comp8b_kl
#           scripts/run_feat215_split.sh hpA <hp1_B python PID>   # GPUs 1,2: nm/hp1_A
#   host B: scripts/run_feat215_split.sh pathwise|bankcap|comma|nm1984a|nm1984b
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
wait_pid () {  # pid expected-argv-fragment: wait, with a deadline, for an arm this lane follows
  ps -o args= -p "$1" | grep -q -- "$2" || { echo "[feat215] PID $1 is not $2"; exit 1; }
  local dl=$(( $(date +%s) + 10800 ))
  while kill -0 "$1" 2>/dev/null; do [ "$(date +%s)" -gt "$dl" ] && { echo "[feat215] deadline on $1"; exit 1; }; sleep 20; done
  echo "[feat215] $2 finished (PID $1) $(date '+%F %T')"
}
case "${1:-}" in
kl)       wait_pid "$2" output/chainfix/phase1; export CUDA_VISIBLE_DEVICES=4
          job comp8b_kl --risky-model $M --limit 100 --k-values -1 0 3 5 10 20 --windows 20 50 ;;
hpA)      wait_pid "$2" output/chainfix/nm/hp1_B; export CUDA_VISIBLE_DEVICES=1,2
          job nm/hp1_A $C --split test --novel harry_potter --limit 50 --temperature 1.0 --repetition-penalty 1.0 --k-values $KS ;;
pathwise) export CUDA_VISIBLE_DEVICES=0
          job comp8b_pathwise --risky-model $M --limit 100 --k-values -1 0 1 3 5 10 20 50 --windows 20 50 --constraint pathwise ;;
bankcap)  export CUDA_VISIBLE_DEVICES=1
          job bank_cap_k10_10 --risky-model $M --limit 100 --k-values 10 --windows 50 --bank-cap 10 --batch-size 16
          job bank_cap_k10_50 --risky-model $M --limit 100 --k-values 10 --windows 50 --bank-cap 50 --batch-size 16
          job bank_cap_k20_20 --risky-model $M --limit 100 --k-values 20 --windows 50 --bank-cap 20 --batch-size 16 ;;
comma)    export CUDA_VISIBLE_DEVICES=2
          job comp_comma7b --safe-model common-pile/comma-v0.1-2t --risky-model output/phase4/memorizing_comma7b \
            --limit 100 --k-values -1 0 0.15 0.5 1 3 5 10 20 --windows 20 50 ;;
nm1984a)  export CUDA_VISIBLE_DEVICES=3,4
          job nm/1984_greedy $C --split attack_train --novel 1984 --limit 8 --temperature 1.0 --repetition-penalty 1.0 --k-values -1 0 --greedy
          job nm/1984_B $C --split attack_train --novel 1984 --limit 8 --temperature 0.7 --repetition-penalty 1.1 --k-values $KS ;;
nm1984b)  export CUDA_VISIBLE_DEVICES=5,6
          job nm/1984_A $C --split attack_train --novel 1984 --limit 8 --temperature 1.0 --repetition-penalty 1.0 --k-values $KS ;;
*) echo "usage: $0 kl|hpA <pid> | pathwise|bankcap|comma|nm1984a|nm1984b"; exit 2 ;;
esac
echo "[feat215] lane $1 drained $(date '+%F %T')"
