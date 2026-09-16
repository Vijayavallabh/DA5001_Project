#!/usr/bin/env bash
# feat-120: the k-grid sweeps for the three BookMIA pairs.
#
# The grid is NOT a parameter of this script. It is the Gutenberg arm's grid verbatim, committed in
# results/onset_prediction_bookmia.md at 14:27 on 2026-09-15 -- while the memorisers were in epoch 1
# and no BookMIA s(x) existed -- precisely so that it cannot be tuned to bracket an answer. If a
# crossing lands at either end of it, that is a finding about the grid and is reported as one
# (caution (g)), with the bootstrap no-crossing fraction.
#
# Every flag below is copied from the Gutenberg run log (output/phase5/gut_onset_kl3m.log,
# gut_onset_pleias_phi.log), not reconstructed -- caution (v). --queries-out is not optional
# decoration: analysis/recheck_violations.py reads it for the per-trajectory invariant recheck.
#
# One queue shell per card running its jobs in order (caution (x)): the sequencing is the shell's
# own and no PID is captured. Never GPU 3 (4 GB T400); CUDA_DEVICE_ORDER is set with
# CUDA_VISIBLE_DEVICES, without which CUDA numbers the T400 as 4.
#
# Usage: scripts/run_bookmia_sweeps.sh <gpu> [pair-key]...
#   pair-key is one of the keys in PAIRS below; with none given, all three run in order.
#
# The out-tags live HERE rather than in a hand-typed command line because a typo in one is silent:
# analysis/onset_gutenberg.py prints "no sweep at ..., skipping" to stderr and scores the other
# two, so a mistyped tag reads as a two-pair result rather than as an error. The tags are pinned
# against CORPORA["bookmia"] by tests/test_bookmia_onset.py.
set -u
GPU="${1:?usage: $0 <gpu> [pair-key]...}"; shift

# key            out-tag      anchor                              memoriser
PAIRS=(
  "kl3m520m    | kl3m520m  | output/phase5/anchor_kl3m-002-520m | output/phase5/memb_kl3m-002-520m"
  "pleias12b   | pleias12b | PleIAs/Pleias-1.2b-Preview         | output/phase5/memb_Pleias-1_2b"
  "phi35       | phi35     | output/phase5/anchor_phi35mini     | output/phase5/memb_phi35mini"
)
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
. scripts/gpu_env.sh   # (cwd is the repo root) strips the stale in-repo driver from LD_LIBRARY_PATH
CORPUS=data/bench/bookmia100_onset100.jsonl

# GATE-BEGIN  (tests/test_bookmia_onset.py extracts between these two sentinels)
# P1 is parameter-free and needs no decoding, so its predictions must be COMMITTED before the
# sweep that measures them exists. That is the whole out-of-sample claim, and "I remembered to
# commit first" is not evidence of it -- git is. This refuses to decode a single token until:
#   1. results/onset_theory_bookmia.csv is tracked AND has no uncommitted diff, and
#   2. the pre-registration quotes the predictions ABOVE its "## Scoring log" line.
# Same discipline as scripts/add_pair.sh, which refuses to run the sweep it sets up.
THEORY=results/onset_theory_bookmia.csv
PREREG=results/onset_prediction_bookmia.md
gate_fail() { echo "GATE REFUSED: $1"; echo "Commit the P1 predictions first; no sweep runs."; exit 2; }
[ -f "$THEORY" ] || gate_fail "$THEORY does not exist (run scripts/run_bookmia_p1.sh)"
git ls-files --error-unmatch "$THEORY" >/dev/null 2>&1 || gate_fail "$THEORY is untracked"
git diff --quiet HEAD -- "$THEORY" || gate_fail "$THEORY has uncommitted changes"
sed '/^## Scoring log/,$d' "$PREREG" | grep -q 'pred onset' \
  || gate_fail "$PREREG does not quote the predictions above its scoring log"
echo "=== gate passed: P1 committed at $(git log -1 --format=%h -- "$THEORY") ==="
# GATE-END
# committed before any BookMIA s(x) existed; the two baselines are mandatory (AGENTS.md)
GRID="-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2"

want=("$@")
for row in "${PAIRS[@]}"; do
  IFS='|' read -r key tag safe risky <<<"$row"
  key="${key// /}"; tag="${tag// /}"; safe="${safe// /}"; risky="${risky// /}"
  if [ ${#want[@]} -gt 0 ]; then
    printf '%s\n' "${want[@]}" | grep -qx "$key" || continue
  fi
  out="output/phase5/fineb_${tag}"
  echo "=== sweep $tag  safe=$safe  risky=$risky  ($(date +%H:%M)) ==="
  set -x
  .venv/bin/python analysis/composition_attack.py \
    --safe-model "$safe" --risky-model "$risky" --corpus-file "$CORPUS" \
    --k-values $GRID --modes single --limit 100 \
    --out "$out" --text-out "$out/composition_extracted.csv" \
    --queries-out "$out/queries.jsonl" || { set +x; echo "FAILED $tag"; exit 1; }
  set +x
done
echo "=== queue done $(date +%H:%M) ==="
