#!/usr/bin/env bash
# feat-180's two missing rungs on HOST B, at the user's instruction (the declared deviation in
# results/onset_prediction_vetting_ladder.md): TinyComma and KL3M-1.7B at L=150, with the registered
# `vet` command unchanged (scripts/run_selfix_vetladder_local.sh), and then the declared host check --
# feat-179 Part A's memoriser baseline with Part A's exact protocol, seed 1234, compared passage by
# passage with the local draw (78/100 at recall >= 0.01).
# Usage (host B): setsid nohup bash scripts/run_vet150_hostb.sh <gpu-a> <gpu-b> > /dev/null 2>&1 < /dev/null &
set -u
GA=${1:?gpu}; GB=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
mkdir -p output/logs ~/v/logs
MEM=output/memorizing_llama8b
selx() {   # selx <gpu> <prefix> <selection_extraction.py args...>
  local G=$1 P=$2; shift 2
  local LOG=output/logs/$P.log
  if [ -s "results/${P}_per_passage.csv" ]; then echo "[$P] already on disk" >> "$LOG"; return; fi
  rm -f ~/v/logs/$P.done ~/v/logs/$P.fail
  echo "[$P] start $(date '+%F %T') on host B GPU $G" >> "$LOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_extraction.py "$@" --prefix "$P" --out results >> "$LOG" 2>&1
  local RC=$?
  echo "[$P] exit=$RC at $(date '+%F %T')" >> "$LOG"
  [ $RC -eq 0 ] && touch ~/v/logs/$P.done || touch ~/v/logs/$P.fail
}
vet() { selx "$1" "vetladder_L$2_$3" --safe-model "$4" --risky-model $MEM \
          --raw-prompt --split test --novel harry_potter --limit 50 --seed-tokens "$2" \
          --max-new-tokens 200 --n-values 1 8 64 --batch-size 8; }
{ vet "$GA" 150 tinycomma jacquelinehe/tinycomma-1.8b-llama3-tokenizer
  selx "$GA" hostcheck_memoriser_n1 --risky-model $MEM --n-values 1 --limit 100 --batch-size 32; } &
vet "$GB" 150 kl3m17b alea-institute/kl3m-003-1.7b &
wait
