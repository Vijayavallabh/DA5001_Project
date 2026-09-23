#!/usr/bin/env bash
# feat-172 stage 3 (results/onset_prediction_offsupport_ladder.md): judge B on the n=256 pools, ONLY
# after the registered gate clears. The gate is checked here, by the same reward_gate the scorer
# uses, and a failure stops the script before any judged number exists: "no n>64 number is read
# until it clears" is kept by construction rather than by not looking.
#
#   selection_scaling.py, the same flags as the reward pass minus --rewards-only: it finds the cache
#     and judges the n-grid 1..256 in one pass, for B2 (Arm A) and B5 (Arm B).
#   order_averaged_h2h.py at n=128 and n=256 (Arm A only, for B1): the committed AlpacaEval
#     binding-budget invocation (scripts/run_mixpow_judge_k.sh) with only --sel-dir, --rewards, --n
#     and --tag changed.
# Usage: run_offonsup_judge.sh <a|b> <gpu>
set -u
ARM=${1:?a|b}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
case "$ARM" in
  a) SEL=output/offsup/sel_anchor256; BASE=output/mixpow/baseline; TAG=offsup; OLD=mixpow_rewards64.csv ;;
  b) SEL=output/onsup/sel_merged; BASE=output/wscope/a_baseline; TAG=onsup; OLD=wscope_rewards64_a.csv ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
LOG=output/logs/offonsup_judge_$ARM.log
MARK="$HOME/v/logs/offonsup_judge_$ARM"
rm -f "$MARK.done" "$MARK.fail"
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"
{
echo "[ofj:$ARM] gate $(date +%H:%M:%S)"
if ! .venv/bin/python - "$TAG" "$OLD" <<'EOF'
import sys
sys.path.insert(0, ".")
from analysis.score_n128 import reward_gate, rows
ok, bad = reward_gate(rows(f"results/{sys.argv[1]}_rewards256.csv"), rows(f"results/{sys.argv[2]}"))
print(f"[ofj] ranks 0-63 == {sys.argv[2]}: {'PASS' if ok else 'FAIL'} "
      f"{'' if ok else (bad if isinstance(bad, str) else bad[:5])}")
sys.exit(0 if ok else 1)
EOF
then echo "[ofj:$ARM] GATE FAILED: INAPPLICABLE, nothing judged"; touch "$MARK.fail"; exit 4; fi
env $E .venv/bin/python analysis/selection_scaling.py \
  --gen-dir "$SEL" --baseline-dir "$BASE" \
  --reward-cache "results/${TAG}_rewards256.csv" --tag "_$TAG" --max-n 256 \
  --judges microsoft/Phi-3.5-mini-instruct --batch-size 16 --out results
RC=$?
echo "[ofj:$ARM] selection_scaling rc=$RC $(date +%H:%M:%S)"
if [ "$ARM" = a ] && [ $RC -eq 0 ]; then
  for N in 128 256; do
    env $E .venv/bin/python analysis/order_averaged_h2h.py --judge microsoft/Phi-3.5-mini-instruct \
      --sel-dir "$SEL" --metered-dir output/mixpow/conc_k10 \
      --anchor-dir output/mixpow/sel_anchor64 --baseline-dir output/mixpow/baseline \
      --rewards "results/${TAG}_rewards256.csv" --data-dir data/bench/alpaca \
      --tag "_offsup_n$N" --seed 7717 --n "$N" --k 1 --out results
    R=$?; echo "[ofj:a] h2h n=$N rc=$R $(date +%H:%M:%S)"; [ $R -eq 0 ] || RC=$R
  done
fi
echo "[ofj:$ARM] DONE rc=$RC $(date +%H:%M:%S)"
} >> "$LOG" 2>&1
[ $RC -eq 0 ] && touch "$MARK.done" || touch "$MARK.fail"
