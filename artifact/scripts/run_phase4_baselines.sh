#!/usr/bin/env bash
# The k=-1 and k=0 arms output/phase4/fine_tc and fine_comma never ran.
#
# Working Rules: "Every experiment that reports a copying or spend metric at some k also reports
# k = -1 (risky model only) and k = 0 (safe model only) on the same prompts and seeds." Those two
# sweeps report six budgets each and carry NEITHER baseline in their own summary, so the nine-pair
# table's strength column had to borrow both from companion runs -- sound only because the companion
# `[ca]` protocol lines are identical character for character, which analysis/onset_table.py asserts.
# This replaces the borrow with the measurement.
#
# EVERY flag is a default of composition_attack.py, because the protocol line the two sweeps printed
# is the line the defaults produce (caution (v) -- read the run log, do not reconstruct):
#   100 passages; target length mean 303/260 tokens; seed 20 tokens raw_prompt=False greedy=False;
#   constraint=kl prefix_debt=True temperature=1.0 rp=1.0 retries=1
# so nothing is passed but the models, the two budgets, the two modes the sweeps used, and --limit.
# In particular --batch-size is LEFT AT ITS DEFAULT of 32: caution (u), batch size is part of the
# seed for a sampled arm, and the k=-1 recall is sampled. Passing one explicitly would be a change.
#
# Output goes to NEW directories. fine_tc/ and fine_comma/ are referenced by results/onset_pairs.tsv
# and read by every onset analysis; they are not ours to overwrite for a baseline.
#
# The cross-check that makes this worth running: the companion runs already give k=-1 as 0.4920
# (TinyComma pair) and 0.7189 (Comma-7B). If these fresh arms reproduce those, the borrow is
# confirmed and the daggers come off. If they do not, that is a finding about the protocol and the
# borrowed numbers stay borrowed until it is understood -- it must NOT be quietly replaced.
#
# One queue shell per card running its jobs in order (caution (x)). GPU defaults to 2, the standing
# rule in AGENTS.md.
set -u
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-2}"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
. scripts/gpu_env.sh   # (cwd is the repo root) strips the stale in-repo driver from LD_LIBRARY_PATH

run () {                       # run <tag> <safe-model> <risky-model>
  tag="$1"; safe="$2"; risky="$3"; out="output/phase4/${tag}_base"
  echo "=== $tag baselines ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python analysis/composition_attack.py \
    --safe-model "$safe" --risky-model "$risky" \
    --k-values -1 0 --modes single oracle --limit 100 \
    --out "$out" --queries-out "$out/queries.jsonl" \
    || { set +x; echo "FAILED $tag"; exit 1; }
  set +x
}

run fine_tc    jacquelinehe/tinycomma-1.8b-llama3-tokenizer output/memorizing_llama8b
run fine_comma common-pile/comma-v0.1-2t                    output/phase4/memorizing_comma7b
echo "=== BASELINES DONE $(date +%H:%M) ==="
