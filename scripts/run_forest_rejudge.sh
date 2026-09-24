#!/usr/bin/env bash
# results/forest_deecho_note.md: every judged row of the forest plot, re-judged on recovered text
# (caution (bc)). Each command is the producing command on record plus --deecho and a tag ending
# _deecho; the committed reward cache is read, never rewritten (checked by hash), so the picks are
# the committed picks and only the text the judge reads changes.
# Usage: bash scripts/run_forest_rejudge.sh <gpu> [sweep ...]   (default: all nine, in order)
set -u
GPU=${1:?gpu}; shift
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
M=output/logs/marks; mkdir -p $M output/logs
P=output/phase5; R=results
# name | gen-dir | baseline-dir | max-n | reward cache | tag | corpus
SWEEPS="audited|$P/sel_anchor64|output/sweep_plain|64|$R/selection_rewards64.csv||data
pleias12b|$P/sel_pleias12b_8|output/sweep_plain|8|$R/selection_rewards_pleias12b.csv|_pleias12b|data
kl3m17b|$P/sel_kl3m17b_8|output/sweep_plain|8|$R/selection_rewards_kl3m17b.csv|_kl3m17b|data
comma7b|$P/sel_comma7b_8|output/sweep_plain|8|$R/selection_rewards_comma7b.csv|_comma7b|data
comma1t|$P/sel_comma1t_8|output/sweep_plain|8|$R/selection_rewards8_comma1t.csv|_comma1t|data
pleias3b|$P/sel_pleias3b_8|output/sweep_plain|8|$R/selection_rewards8_pleias3b.csv|_pleias3b|data
alpaca|$P/alpaca_anchor8|$P/alpaca_risky|8|$R/selection_rewards_alpaca.csv|_alpaca|data/bench/alpaca
alpaca_comma7b|$P/alpaca_comma7b_8|$P/alpaca_risky|8|$R/selection_rewards_alpaca_comma7b.csv|_alpaca_comma7b|data/bench/alpaca
mtbench|$P/mtbench_anchor8|$P/mtbench_risky|8|$R/selection_rewards_mtbench.csv|_mtbench|data/bench/mtbench"
WANT=" ${*:-audited pleias12b kl3m17b comma7b comma1t pleias3b alpaca alpaca_comma7b mtbench} "
echo "$SWEEPS" | while IFS='|' read -r name gen base maxn cache tag corpus; do
  case "$WANT" in *" $name "*) ;; *) continue ;; esac
  log=output/logs/forest_rejudge_$name.log
  rm -f $M/forest_$name.done $M/forest_$name.fail
  [ -f "$cache" ] || { echo "missing $cache" >> "$log"; touch $M/forest_$name.fail; continue; }
  h0=$(sha256sum "$cache" | cut -c1-64)
  echo "[forest:$name] start $(date '+%F %T') gpu=$GPU" >> "$log"
  if .venv/bin/python analysis/selection_scaling.py --deecho --data-dir "$corpus" --gen-dir "$gen" \
       --baseline-dir "$base" --max-n "$maxn" --reward-cache "$cache" --tag "${tag}_deecho" \
       --out results < /dev/null >> "$log" 2>&1 && [ "$(sha256sum "$cache" | cut -c1-64)" = "$h0" ]; then
    touch $M/forest_$name.done
  else
    touch $M/forest_$name.fail
  fi
  echo "[forest:$name] end $(date '+%F %T')" >> "$log"
done
