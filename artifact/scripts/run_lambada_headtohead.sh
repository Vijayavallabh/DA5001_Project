#!/usr/bin/env bash
# feat-154: the second judge-free head-to-head, on LAMBADA.
# Bands: results/onset_prediction_lambada_headtohead.md, committed before this ran.
set -uo pipefail
source "$HOME/v/env.sh"
# Derived, not hardcoded: a literal `$HOME/v/<project>` puts the repository name inside
# every launcher, and scripts/build_artifact.sh refuses to ship a tree containing it.
cd "$(dirname "$0")/.."
CARDS="${1:-2}"
MARK="$HOME/v/logs/lambada"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CARDS"
export HF_HUB_CACHE="$PWD/hf_cache"
set -a; source .env 2>/dev/null; set +a
{
echo "[lambada] build corpus $(date +%H:%M)"
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from analysis.build_bench_corpora import build_lambada; build_lambada()"
export HF_HUB_OFFLINE=1
COMMON="--data-dir data/bench/lambada --safe-model-path jacquelinehe/tinycomma-1.8b-llama3-tokenizer --risky-model-path meta-llama/Meta-Llama-3.1-8B-Instruct --cap-factual 500 --cap-neutral 0 --cap-creative 0 --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 8 --batch-size 48"
echo "[lambada] metered $(date +%H:%M)"
.venv/bin/python h1.py --k-values -1 0 0.5 1 3 20 --trajectories-per-prompt 1 $COMMON \
  --output-dir output/phase5/lambada_metered
echo "[lambada] metered exit=$?"
echo "[lambada] selection $(date +%H:%M)"
.venv/bin/python h1.py --k-values 0 --trajectories-per-prompt 64 $COMMON \
  --output-dir output/phase5/lambada_sel64
echo "[lambada] selection exit=$?"
echo "[lambada] score $(date +%H:%M)"
.venv/bin/python analysis/verifiable_metered.py --task lambada \
  --metered-dir output/phase5/lambada_metered --selection-dir output/phase5/lambada_sel64 \
  --corpus data/bench/lambada_factual.jsonl --tag _lambada --out results
echo "[lambada] score exit=$?"
} > "${MARK}.log" 2>&1
if [ -f results/verifiable_metered_lambada.csv ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
