#!/usr/bin/env bash
# Plan v5: register one (anchor, risky) pair with the onset pipeline and refresh everything.
#
# Adding a pair means touching two TSV manifests, one budget-path run and three analyses, in an
# order that matters: the prediction must be computed and committed BEFORE the sweep is read, or the
# pre-registration is worthless. This script does the registration and refresh; it deliberately does
# NOT run the sweep, so the human step of committing the prediction sits between the two.
#
# Usage:
#   scripts/add_pair.sh "<label>" <anchor-model-id> <memoriser-dir> <composition_summary.csv> [gpu]
# Example:
#   scripts/add_pair.sh "Pleias-350M + mem. Pleias-350M" PleIAs/Pleias-350m-Preview \
#     output/phase5/mem_Pleias-350m-Preview output/phase5/fine_pleias350m/composition_summary.csv 4
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL=${1:?label}; ANCHOR=${2:?anchor model id}; RISKY=${3:?memoriser dir}
COMP=${4:?composition_summary.csv}; GPU=${5:-4}
TAG=$(echo "$LABEL" | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_.-')
BP="results/budget_path_${TAG}.csv"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1
export HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python

if [ ! -f "$BP" ]; then
  echo "[add_pair] budget path for $ANCHOR -> $BP"
  $PY analysis/budget_path.py --safe-model "$ANCHOR" --composition '' --limit 100 \
      --out results --prefix "budget_path_${TAG}"
fi

# theory manifest: 4th column (measured onset) left empty -> prediction only
if ! grep -qF "$LABEL" results/onset_theory_pairs.tsv 2>/dev/null; then
  printf '%s\t%s\t%s\t\n' "$LABEL" "$RISKY" "$BP" >> results/onset_theory_pairs.tsv
  echo "[add_pair] appended to results/onset_theory_pairs.tsv (prediction only)"
fi

# onset manifest: only once the sweep exists, otherwise analysis/onset.py has nothing to read
if [ -f "$COMP" ] && ! grep -qF "$LABEL" results/onset_pairs.tsv 2>/dev/null; then
  # 5 columns: label, sweep summary, budget path, tokenizer (analysis/onset_units.py) and the
  # pair's name in onset_theory_per_work.csv (analysis/collapse_robustness.py). add_pair.sh uses
  # one LABEL for both manifests, so the fifth field is the label itself.
  printf '%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$COMP" "$BP" "$ANCHOR" "$LABEL" >> results/onset_pairs.tsv
  echo "[add_pair] appended to results/onset_pairs.tsv"
elif [ ! -f "$COMP" ]; then
  echo "[add_pair] no sweep at $COMP yet -- prediction registered, measurement pending."
  echo "[add_pair] COMMIT THE PREDICTION NOW, then run the sweep, then re-run this script."
fi

$PY analysis/onset_theory.py --out results
[ -f "$COMP" ] && $PY analysis/onset.py --out results --thresh 0.01
if [ -f "$COMP" ]; then
  # onset_ci needs the per-passage file and the pair's own s(x); a bare invocation is an
  # argparse error, and with `set -e` that would abort the whole refresh.
  SX=$($PY -c "import csv,statistics as st;print(st.median(float(r['s_mean']) for r in csv.DictReader(open('$BP'))))")
  $PY analysis/onset_ci.py --comp "$(dirname "$COMP")/composition.csv" --s-x "$SX" --label "$LABEL" --out results
fi
[ -f "$COMP" ] && $PY analysis/onset_units.py --out results
[ -f "$COMP" ] && $PY analysis/collapse_robustness.py --out results
[ -f "$COMP" ] && $PY analysis/onset_ladder.py --out results
[ -f "$COMP" ] && $PY figures/make_figures_v4.py \
  --copy-to "${SATML_DIR:-../sub/satml}/figures"
echo "[add_pair] done"
