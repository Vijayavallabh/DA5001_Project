#!/usr/bin/env bash
# feat-216 (results/onset_prediction_tempering.md): the anchor alone at three sampling laws, judged into the
# committed feat-210 pass. Local A100s only (the judging machine type of that pass).
#   GPU=<i>   bash scripts/run_feat216.sh gen <t07|t05|t07pen>
#   GPUS=<ij> bash scripts/run_feat216.sh judge <B|G> <t07|t05|t07pen>...   (new arms into output/feat216/judge)
#             bash scripts/run_feat216.sh final <B|G>                      (merge caches, every arm cached)
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
L=output/logs; O=output/feat216; J=$O/judge
mkdir -p $L $O $J
H1="h1.py --k-values 0 --trajectories-per-prompt 1 --seeds 52 --cap-neutral 200 --cap-factual 150 --cap-creative 150
    --cap-val 0 --cap-test 0 --cap-attack-train 0 --batch-size 48 --trust-remote-code"
judge_args() { case "$1" in
  B) echo "--judge microsoft/Phi-3.5-mini-instruct" ;;
  G) echo "--judge google/gemma-2-27b-it --device-map auto" ;; esac; }
case "${1:?mode}" in
  gen)
    export CUDA_VISIBLE_DEVICES=${GPU:?GPU}
    case "$2" in
      t07) S="--temperature 0.7 --repetition-penalty 1.0" ;;
      t05) S="--temperature 0.5 --repetition-penalty 1.0" ;;
      t07pen) S="--temperature 0.7 --repetition-penalty 1.1" ;;
      *) echo "unknown arm $2"; exit 2 ;;
    esac
    if $PY $H1 $S --output-dir $O/$2 > $L/feat216_gen_$2.log 2>&1; then touch $L/feat216_gen_$2.done
    else touch $L/feat216_gen_$2.fail; fi ;;
  judge)
    export CUDA_VISIBLE_DEVICES=${GPUS:?GPUS}
    jd=$2; shift 2
    for arm in "$@"; do
      if $PY analysis/matched_h2h.py --tag tempering_${jd}_$arm --out $J $(judge_args $jd) \
           --arm anchor_$arm=arm:$O/$arm:0:kl > $L/feat216_judge${jd}_$arm.log 2>&1
      then touch $L/feat216_judge${jd}_$arm.done; else touch $L/feat216_judge${jd}_$arm.fail; fi
    done ;;
  final)
    jd=$2
    $PY - "$jd" <<'PY'
import csv, sys
jd = sys.argv[1]
rows = [r for r in csv.DictReader(open(f"results/matched_h2h_verdicts_matched_plain_{jd}.csv"))
        if r["arm"] in ("sel_n64", "sel_n1", "anchor_k0")]
for arm in ("t07", "t05", "t07pen"):
    rows += list(csv.DictReader(open(f"output/feat216/judge/matched_h2h_verdicts_tempering_{jd}_{arm}.csv")))
with open(f"results/matched_h2h_verdicts_tempering_{jd}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["arm", "prompt_id", "first", "second", "text_sha"])
    w.writeheader(); w.writerows(rows)
print("merged", len(rows), "verdict rows")
PY
    export CUDA_VISIBLE_DEVICES=""
    $PY analysis/matched_h2h.py --tag tempering_$jd $(judge_args $jd) \
      --arm sel_n64=sel:64 --arm sel_n1=sel:1 --arm anchor_k0=arm:output/sweep_plain:0:kl \
      --arm anchor_t07=arm:$O/t07:0:kl --arm anchor_t05=arm:$O/t05:0:kl --arm anchor_t07pen=arm:$O/t07pen:0:kl \
      --control sel_n64=sel_n1 --control anchor_t07=sel_n1 --control anchor_t05=sel_n1 \
      --control anchor_t07pen=sel_n1 --control anchor_k0=sel_n1 \
      --diff sel_n64,anchor_t07 --diff sel_n64,anchor_t05 --diff sel_n64,anchor_t07pen ;;
esac
