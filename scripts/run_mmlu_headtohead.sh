#!/usr/bin/env bash
# feat-147: the second judge-free head-to-head, on MMLU.
# Bands: results/onset_prediction_mmlu_headtohead.md, committed before this ran.
set -uo pipefail
source "$HOME/v/env.sh"
# Derived, not hardcoded: a literal `$HOME/v/<project>` puts the repository name inside
# every launcher, and scripts/build_artifact.sh refuses to ship a tree containing it.
cd "$(dirname "$0")/.."
CARDS="${1:-2}"
MARK="$HOME/v/logs/mmlu"
rm -f "${MARK}.done" "${MARK}.fail"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CARDS"
export HF_HUB_CACHE="$PWD/hf_cache"
set -a; source .env 2>/dev/null; set +a
{
echo "[mmlu] build corpus $(date +%H:%M)"
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from analysis.build_bench_corpora import build_mmlu; build_mmlu()"
export HF_HUB_OFFLINE=1
COMMON="--data-dir data/bench/mmlu --safe-model-path jacquelinehe/tinycomma-1.8b-llama3-tokenizer --risky-model-path meta-llama/Meta-Llama-3.1-8B-Instruct --cap-factual 500 --cap-neutral 0 --cap-creative 0 --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 24 --batch-size 48"
echo "[mmlu] metered $(date +%H:%M)"
.venv/bin/python h1.py --k-values -1 0 0.5 1 3 20 --trajectories-per-prompt 1 $COMMON \
  --output-dir output/phase5/mmlu_metered
echo "[mmlu] metered exit=$?"
echo "[mmlu] selection $(date +%H:%M)"
.venv/bin/python h1.py --k-values 0 --trajectories-per-prompt 64 $COMMON \
  --output-dir output/phase5/mmlu_sel64
echo "[mmlu] selection exit=$?"
echo "[mmlu] score $(date +%H:%M)"
.venv/bin/python analysis/verifiable_metered.py --task mmlu \
  --metered-dir output/phase5/mmlu_metered --selection-dir output/phase5/mmlu_sel64 \
  --corpus data/bench/mmlu_factual.jsonl --tag _mmlu --out results
echo "[mmlu] score exit=$?"
} > "${MARK}.log" 2>&1
if [ -f results/verifiable_metered_mmlu.csv ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
