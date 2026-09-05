#!/usr/bin/env bash
# feat-018: natural-memorisation audit with the mechanism's own risky model (Llama 3.1 70B base, unsloth mirror, bf16 sharded
# over GPUs 1+2 with TinyComma resident) on the CopyBench passages that Cooper et al. report as memorised (Harry Potter 1, test
# split; 1984, attack split). Two decoding settings: A = temperature 1, no penalty (the released logs); B = temperature 0.7,
# repetition penalty 1.1 (He et al. App. D.1). Every run logs one JSON line per query (spend, budget, realised ratio, activity).
# Usage: scripts/run_natural_memorisation.sh [batch_size]   (run in the background; ~hours)
set -u
cd "$(dirname "$0")/.."
SNAP=hf_cache/models--unsloth--Meta-Llama-3.1-70B/snapshots/1b7306651142d0cc65d993076a250a6a82cf046c
BS=${1:-16}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 HF_HUB_OFFLINE=1
KS="-1 0 1 1.5 3 5 10 20"
run () {  # name split novel limit temperature rp extra...
  local name=$1 split=$2 novel=$3 limit=$4 temp=$5 rp=$6; shift 6
  mkdir -p output/phase2/nm/$name
  .venv/bin/python analysis/composition_attack.py --risky-model $SNAP --risky-device-map auto --max-memory "0=72GiB,1=64GiB" \
    --split $split --novel "$novel" --limit $limit --k-values $KS --windows 50 --modes single oracle chained \
    --temperature $temp --repetition-penalty $rp --batch-size $BS "$@" \
    --out output/phase2/nm/$name --figures output/phase2/nm/$name --text-out output/phase2/nm/$name/extracted.csv \
    --queries-out output/phase2/nm/$name/queries.jsonl > output/phase2/nm/$name/run.log 2>&1
  echo "$name exit $? $(date +%H:%M)"
}
run hp1_B test harry_potter 50 0.7 1.1
run hp1_A test harry_potter 50 1.0 1.0
run 1984_B attack_train 1984 8 0.7 1.1
run 1984_A attack_train 1984 8 1.0 1.0
run hp1_B_pathwise test harry_potter 50 0.7 1.1 --constraint pathwise
run hp1_B_nodebt test harry_potter 50 0.7 1.1 --no-prefix-debt
echo "all natural-memorisation runs finished $(date)"
