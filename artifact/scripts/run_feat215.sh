#!/usr/bin/env bash
# feat-215 (results/onset_prediction_chained_fix.md): every committed chained arm re-run with the left-pad slicing
# fixed (caution (bc)). Each job is the original command with --modes chained and only the output paths changed:
# same models, passages, budgets, windows, seed (default 1234), batch size and decoding settings (each original
# run.log's protocol line). Local A100s, the hardware the originals ran on.
#   scripts/run_feat215.sh small   # GPU 4: the 8B memoriser arms, the three bank caps, the Comma-7B pair
#   scripts/run_feat215.sh nm      # GPUs 1,2: the six 70B natural-memorisation arms
set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
O=output/chainfix
job () {  # name args...
  local name=$1; shift
  local d=$O/$name; mkdir -p $d
  .venv/bin/python analysis/composition_attack.py --modes chained "$@" \
    --out $d --figures $d --text-out $d/extracted.csv --queries-out $d/queries.jsonl > $d/run.log 2>&1
  echo "[feat215] $name exit $? $(date '+%F %T')"
}
case "${1:-}" in
small)
  export CUDA_VISIBLE_DEVICES=4
  M=output/memorizing_llama8b
  job phase1 --risky-model $M --limit 100 --k-values -1 0 0.15 0.5 1 3 5 10 20 --windows 20 50
  job comp8b_kl --risky-model $M --limit 100 --k-values -1 0 3 5 10 20 --windows 20 50
  job comp8b_pathwise --risky-model $M --limit 100 --k-values -1 0 1 3 5 10 20 50 --windows 20 50 --constraint pathwise
  job bank_cap_k10_10 --risky-model $M --limit 100 --k-values 10 --windows 50 --bank-cap 10 --batch-size 16
  job bank_cap_k10_50 --risky-model $M --limit 100 --k-values 10 --windows 50 --bank-cap 50 --batch-size 16
  job bank_cap_k20_20 --risky-model $M --limit 100 --k-values 20 --windows 50 --bank-cap 20 --batch-size 16
  job comp_comma7b --safe-model common-pile/comma-v0.1-2t --risky-model output/phase4/memorizing_comma7b \
    --limit 100 --k-values -1 0 0.15 0.5 1 3 5 10 20 --windows 20 50
  ;;
nm)
  export CUDA_VISIBLE_DEVICES=1,2
  SNAP=hf_cache/models--unsloth--Meta-Llama-3.1-70B/snapshots/1b7306651142d0cc65d993076a250a6a82cf046c
  C="--risky-model $SNAP --risky-device-map auto --max-memory 0=75GiB,1=70GiB --raw-prompt --seed-tokens 100 --windows 50 --batch-size 16"
  KS="-1 0 1 1.5 3 5 10 20"
  job nm/hp1_greedy $C --split test --novel harry_potter --limit 50 --temperature 1.0 --repetition-penalty 1.0 --k-values -1 0 --greedy
  job nm/hp1_B $C --split test --novel harry_potter --limit 50 --temperature 0.7 --repetition-penalty 1.1 --k-values $KS
  job nm/hp1_A $C --split test --novel harry_potter --limit 50 --temperature 1.0 --repetition-penalty 1.0 --k-values $KS
  job nm/1984_greedy $C --split attack_train --novel 1984 --limit 8 --temperature 1.0 --repetition-penalty 1.0 --k-values -1 0 --greedy
  job nm/1984_B $C --split attack_train --novel 1984 --limit 8 --temperature 0.7 --repetition-penalty 1.1 --k-values $KS
  job nm/1984_A $C --split attack_train --novel 1984 --limit 8 --temperature 1.0 --repetition-penalty 1.0 --k-values $KS
  ;;
*) echo "usage: $0 small|nm"; exit 2 ;;
esac
echo "[feat215] queue $1 drained $(date '+%F %T')"
