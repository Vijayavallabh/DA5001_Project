#!/usr/bin/env bash
# The k=-1 and k=0 arms output/phase4/renyi_renyi_{1_0,2,4,8} never ran.
#
# Those four sweep k in {1,3,5} on single and oracle L=50 and feed Table 3, the four-order
# comparison, with NEITHER baseline -- the last of the six real cases a scan of every sweep on disk
# turned up, after fine_tc and fine_comma were measured earlier today.
#
# The prediction is committed in results/onset_prediction_renyi_baselines.md BEFORE this runs: all
# four orders must reproduce output/phase4/fine_tc_base exactly, because at k=-1 the decoder serves
# the risky model alone and at k=0 the safe model alone, and a_patch/factory.py reads
# self.constraint only in the branch that solves under a budget. Running it per order is the point:
# the four arms are the check, not a formality.
#
# Every flag is a default except those that reproduce each sweep's own protocol line --- the models,
# the two budgets, the modes and window the sweeps used, and --constraint. In particular
# --batch-size is LEFT AT ITS DEFAULT of 32: caution (u), batch size is part of the seed for a
# sampled arm, and these baselines are sampled.
#
# --text-out is given explicitly. Its default is one fixed path inside output/composition/, so every
# run that omits it appends a burst to a directory whose own job ended on 2026-09-05 and drags that
# row's end time forward; on 2026-09-16 that inflated the project total by 110 GPU-hours before
# analysis/compute_hours.py was fixed to remove every idle gap rather than only the largest.
#
# One queue shell per card (caution (x)). GPU defaults to 2, the standing rule in AGENTS.md.
set -u
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-2}"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
. scripts/gpu_env.sh   # (cwd is the repo root) strips the stale in-repo driver from LD_LIBRARY_PATH

SAFE=jacquelinehe/tinycomma-1.8b-llama3-tokenizer
RISKY=output/memorizing_llama8b

for spec in "1_0:renyi:1.0" "2:renyi:2" "4:renyi:4" "8:renyi:8"; do
  tag="${spec%%:*}"; con="${spec#*:}"
  out="output/phase4/renyi_renyi_${tag}_base"
  echo "=== renyi_${tag} baselines, constraint=${con} ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python analysis/composition_attack.py \
    --safe-model "$SAFE" --risky-model "$RISKY" --constraint "$con" \
    --k-values -1 0 --modes single oracle --windows 50 --limit 100 \
    --out "$out" --text-out "$out/composition_extracted.csv" \
    --queries-out "$out/queries.jsonl" \
    || { set +x; echo "FAILED renyi_${tag}"; exit 1; }
  set +x
done
echo "=== RENYI BASELINES DONE $(date +%H:%M) ==="
