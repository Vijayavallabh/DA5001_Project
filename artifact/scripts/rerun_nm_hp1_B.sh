#!/usr/bin/env bash
# feat-018: rerun hp1_B (Harry Potter test split, He et al.'s book decoding settings) after scripts/run_natural_memorisation.sh
# finishes. The first attempt was stopped at k=0 because the decoder refused warpers under a budget (fixed 2026-09-06).
set -u
cd "$(dirname "$0")/.."
while pgrep -f run_natural_memorisation.sh > /dev/null; do sleep 300; done
SNAP=hf_cache/models--unsloth--Meta-Llama-3.1-70B/snapshots/1b7306651142d0cc65d993076a250a6a82cf046c
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 HF_HUB_OFFLINE=1
d=output/phase2/nm/hp1_B; mkdir -p $d
.venv/bin/python analysis/composition_attack.py --risky-model $SNAP --risky-device-map auto --max-memory 0=75GiB,1=70GiB --raw-prompt --seed-tokens 100 --windows 50 --batch-size 16 \
  --split test --novel harry_potter --limit 50 --temperature 0.7 --repetition-penalty 1.1 --k-values -1 0 1 1.5 3 5 10 20 --modes single oracle chained \
  --out $d --figures $d --text-out $d/extracted.csv --queries-out $d/queries.jsonl > $d/run.log 2>&1
echo "hp1_B exit $? $(date +%H:%M)"
