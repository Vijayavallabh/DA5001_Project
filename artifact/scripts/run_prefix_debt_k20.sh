#!/usr/bin/env bash
# feat-032: the one row missing from results/prefix_debt_ablation.csv. The phase-2 no-prefix-debt run on the 70B
# (scripts/run_natural_memorisation.sh, hp1_B_nodebt) stopped at k=10, so the paper's strongest ablation does not
# cover k=20 -- the top of the range He et al. sweep, and the budget at which the decoder is the identity.
# Same model, settings, passages, seeds and window length as hp1_B_nodebt; only --k-values differs.
# Needs TWO GPUs for the 70B in bf16. Usage: scripts/run_prefix_debt_k20.sh [batch_size]
set -u
cd "$(dirname "$0")/.."
SNAP=hf_cache/models--unsloth--Meta-Llama-3.1-70B/snapshots/1b7306651142d0cc65d993076a250a6a82cf046c
BS=${1:-16}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 HF_HUB_OFFLINE=1
NAME=hp1_B_nodebt_k20
mkdir -p output/phase3/nm/$NAME
.venv/bin/python analysis/composition_attack.py \
  --risky-model $SNAP --risky-device-map auto --max-memory 0=75GiB,1=70GiB \
  --raw-prompt --seed-tokens 100 --windows 50 --batch-size $BS \
  --split test --novel harry_potter --limit 50 --temperature 0.7 --repetition-penalty 1.1 \
  --k-values 20 --modes single oracle --no-prefix-debt \
  --out output/phase3/nm/$NAME --figures output/phase3/nm/$NAME \
  --text-out output/phase3/nm/$NAME/extracted.csv \
  --queries-out output/phase3/nm/$NAME/queries.jsonl > output/phase3/nm/$NAME/run.log 2>&1
echo "$NAME exit $? $(date +%H:%M)"
