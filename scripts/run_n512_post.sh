#!/usr/bin/env bash
# feat-181 (results/onset_prediction_n512_ladder.md), HOST B, after every extension job is .done:
#   merge + G2  ->  reward pass (--rewards-only, batch 16 as committed)  ->  G1  ->  judge B to 512
#   ->  (Arm A) the order-averaged head-to-head at n=512.
# Each gate stops the chain on failure, so no judged number exists unless G1 and G2 passed (G0 was
# read before any extension draw). Commands are feat-172's (scripts/run_offonsup_{rewards,judge}.sh)
# with only the pool, --max-n, the cache and the tag changed.
# Usage: run_n512_post.sh <a|b> <gpu>
set -u
ARM=${1:?a|b}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
M=output/n512/${ARM}_merged
case "$ARM" in
  a) BASE=output/mixpow/baseline; OLD=results/offsup_rewards256.csv
     MERGE=("--classes factual --src output/offsup/sel_anchor256 $(for s in $(seq 256 32 480); do printf 'output/n512/a_s%s_c32 ' $s; done)") ;;
  b) BASE=output/wscope/a_baseline; OLD=results/onsup_rewards256.csv
     MERGE=("--classes neutral creative --src output/onsup/sel_small output/n512/small_s256_c128 output/n512/small_s384_c128"
            "--classes factual --src output/onsup/sel_factual output/n512/factual_s256_c43 output/n512/factual_s299_c43 output/n512/factual_s342_c43 output/n512/factual_s385_c43 output/n512/factual_s428_c42 output/n512/factual_s470_c42") ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
TAG=n512$ARM
CACHE=results/${TAG}_rewards512.csv
LOG=output/logs/n512_post_$ARM.log
MARK=~/v/logs/n512_post_$ARM
rm -f "$MARK.done" "$MARK.fail"
E="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache"
fail() { echo "[post:$ARM] $1 $(date '+%F %T')" >> "$LOG"; touch "$MARK.fail"; exit "${2:-4}"; }
{
echo "[post:$ARM] merge $(date '+%F %T')"
for spec in "${MERGE[@]}"; do
  # shellcheck disable=SC2086
  .venv/bin/python analysis/n512_pool.py merge --out "$M" $spec || fail "G2 FAILED: pool incomplete, nothing scored"
done
if [ -e "$CACHE" ]; then echo "[post:$ARM] $CACHE exists; not overwritten (caution (ax))"
else
  env $E .venv/bin/python analysis/selection_scaling.py --gen-dir "$M" --baseline-dir "$BASE" \
    --reward-cache "$CACHE" --tag "_$TAG" --max-n 512 \
    --judges microsoft/Phi-3.5-mini-instruct --batch-size 16 --rewards-only --out results \
    || fail "reward pass failed" 5
fi
.venv/bin/python - "$CACHE" "$OLD" <<'EOF' || fail "G1 FAILED: INAPPLICABLE, nothing judged"
import sys
sys.path.insert(0, ".")
from analysis.score_n128 import reward_gate, rows
ok, bad = reward_gate(rows(sys.argv[1]), rows(sys.argv[2]), top=256)
print(f"[post] G1 ranks 0-255 == {sys.argv[2]}: {'PASS' if ok else 'FAIL'} {'' if ok else bad[:5]}")
sys.exit(0 if ok else 1)
EOF
env $E .venv/bin/python analysis/selection_scaling.py --gen-dir "$M" --baseline-dir "$BASE" \
  --reward-cache "$CACHE" --tag "_$TAG" --max-n 512 \
  --judges microsoft/Phi-3.5-mini-instruct --batch-size 16 --out results || fail "judge failed" 6
if [ "$ARM" = a ]; then
  env $E .venv/bin/python analysis/order_averaged_h2h.py --judge microsoft/Phi-3.5-mini-instruct \
    --sel-dir "$M" --metered-dir output/mixpow/conc_k10 \
    --anchor-dir output/mixpow/sel_anchor64 --baseline-dir output/mixpow/baseline \
    --rewards "$CACHE" --data-dir data/bench/alpaca \
    --tag "_${TAG}_n512" --seed 7717 --n 512 --k 1 --out results || fail "head-to-head failed" 7
fi
echo "[post:$ARM] DONE $(date '+%F %T')"
} >> "$LOG" 2>&1
touch "$MARK.done"
