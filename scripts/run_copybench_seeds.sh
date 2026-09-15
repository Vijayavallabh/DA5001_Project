#!/usr/bin/env bash
# The seed test on the table the claim is actually about (results/onset_prediction_seedspread_copybench.md).
#
# KL3M-520M + mem. KL3M-520M, the pair that is IN the nine-pair table, on the corpus that table is
# built from. Every flag is recovered from output/phase5/mem_kl3m-002-520m/recipe.json (the
# fine-tune) and PROVEN against output/phase5/fine_kl3m520m/composition.csv (the corpus): rebuilding
# --split attack_train --limit 100 reproduces the swept prompt ids exactly and in order. The
# original command survives in no log, which is caution (v)'s situation; recipe.json plus a verified
# reconstruction is what replaces it.
#
# The grid is the pair's OWN committed grid, read off its summary, not chosen.
#
# One queue shell per card (caution (x)). GPU defaults to 2 -- the standing rule in AGENTS.md --
# and is overridden only while the user's multi-card window is open.
set -u
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-2}"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"

BASE=alea-institute/kl3m-002-520m
GRID="-1 0 1.6 1.8 2.0 2.1 2.2 2.3 2.4 2.6 2.8 3.0 3.4"

for sd in "$@"; do
  mem="output/phase5/memc_kl3m520m_s${sd}"
  out="output/phase5/finec_kl3m520m_s${sd}"
  echo "=== copybench seed=$sd  memoriser ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python recipes/finetune_memorizing.py \
    --base "$BASE" --tokenizer "$BASE" --data data --splits attack_train val \
    --target-modules all-linear --no-chat --epochs 40 --seed "$sd" --lr 3e-4 --rank 128 \
    --batch 2 --accum 4 --max-len 0 --stop-loss 0.02 --out "$mem" \
    || { set +x; echo "FAILED finetune seed=$sd"; exit 1; }
  set +x
  echo "=== copybench seed=$sd  sweep ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python analysis/composition_attack.py \
    --safe-model "$BASE" --risky-model "$mem" --split attack_train --limit 100 \
    --k-values $GRID --modes single \
    --out "$out" --text-out "$out/composition_extracted.csv" \
    --queries-out "$out/queries.jsonl" || { set +x; echo "FAILED sweep seed=$sd"; exit 1; }
  set +x
done
echo "=== copybench seed queue done $(date +%H:%M) ==="
