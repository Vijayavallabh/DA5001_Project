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

# CBPAIR selects the table row. Each row's hyperparameters are ITS OWN, read from that pair's
# recipe.json -- the nine memorisers in the table are NOT matched: KL3M-520M trained at batch 2 /
# accum 4 / stop-loss 0.02, Pleias-1.2B at batch 4 / accum 2 / stop-loss 0.03. A ladder must
# inherit its corner's recipe or it is not a ladder on that corner at all.
CBPAIR="${CBPAIR:-kl3m520m}"
case "$CBPAIR" in
  # MAXLEN: 0 means auto = max(token length). For KL3M-520M auto reproduces the corner's recorded
  # 679 exactly, so 0 is right. For Pleias-1.2B auto gives 342 while its corner records 448, so the
  # corner was given an explicit --max-len and we pass the same. Nothing is truncated at either
  # value (max(lens)=342 < 448) and padding is dynamic per batch, so the training is identical --
  # this matches the recorded field so a later reader diffing recipe.json sees no discrepancy.
  # SAFE is the SWEEP's safe model and is NOT always BASE. kl3m-002-520m is published only as
  # pytorch_model.bin; the factory loads with use_safetensors=True, so the sweep must use the
  # materialised copy (scripts/materialise_anchor.py) while the fine-tune uses the fp32 HF id, as
  # the table's own recipe.json records. The two are the same model: identical tensor names and
  # shapes, differing only at bf16 precision (max |diff| 8.6e-4, relative 1.9e-3), verified
  # bitwise. Taking --safe-model from recipe.json's `base` cost two aborted sweeps on 2026-09-16.
  kl3m520m) BASE=alea-institute/kl3m-002-520m; SAFE=output/phase5/anchor_kl3m-002-520m
            PFX=memc_kl3m520m; SFX=finec_kl3m520m
            BATCH=2; ACCUM=4; STOP=0.02; MAXLEN=0
            GRID="-1 0 1.6 1.8 2.0 2.1 2.2 2.3 2.4 2.6 2.8 3.0 3.4" ;;
  pleias12b) BASE=PleIAs/Pleias-1.2b-Preview; SAFE=PleIAs/Pleias-1.2b-Preview
            PFX=memc_pleias12b; SFX=finec_pleias12b
            BATCH=4; ACCUM=2; STOP=0.03; MAXLEN=448
            GRID="-1 0 2 2.4 2.6 2.7 2.8 2.9 3 3.2 3.6" ;;
  *) echo "CBPAIR must be kl3m520m or pleias12b"; exit 2 ;;
esac

for sd in "$@"; do
  mem="output/phase5/${PFX}_s${sd}"
  out="output/phase5/${SFX}_s${sd}"
  echo "=== $CBPAIR seed=$sd  memoriser ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python recipes/finetune_memorizing.py \
    --base "$BASE" --tokenizer "$BASE" --data data --splits attack_train val \
    --target-modules all-linear --no-chat --epochs 40 --seed "$sd" --lr 3e-4 --rank 128 \
    --batch "$BATCH" --accum "$ACCUM" --max-len "$MAXLEN" --stop-loss "$STOP" --out "$mem" \
    || { set +x; echo "FAILED finetune seed=$sd"; exit 1; }
  set +x
  echo "=== $CBPAIR seed=$sd  sweep ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python analysis/composition_attack.py \
    --safe-model "$SAFE" --risky-model "$mem" --split attack_train --limit 100 \
    --k-values $GRID --modes single \
    --out "$out" --text-out "$out/composition_extracted.csv" \
    --queries-out "$out/queries.jsonl" || { set +x; echo "FAILED sweep seed=$sd"; exit 1; }
  set +x
done
echo "=== copybench seed queue done $(date +%H:%M) ==="
