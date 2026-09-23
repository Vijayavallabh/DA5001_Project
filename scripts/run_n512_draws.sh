#!/usr/bin/env bash
# feat-181 (results/onset_prediction_n512_ladder.md): draw trajectory indices [START, START+COUNT) for
# one of feat-172's three processes, on HOST B, with every flag of the committed launcher
# (scripts/run_offsupport.sh, scripts/run_onsupport.sh) unchanged. --trajectory-start is the only
# addition; tests/test_seeds.py pins that a tail run gives exactly the draws a full run would.
#
# COUNT=1 at START=255 is gate G0: it regenerates a committed draw, which must come back byte-identical.
# Usage: run_n512_draws.sh <a|small|factual> <start> <count> <gpu>
set -u
ARM=${1:?a|small|factual}; START=${2:?start}; COUNT=${3:?count}; GPU=${4:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs output/n512 ~/v/logs
set -a; . ./.env; set +a
case "$ARM" in
  a)       C="--cap-factual 805 --cap-neutral 0 --cap-creative 0"; D=data/bench/alpaca ;;
  small)   C="--cap-neutral 200 --cap-creative 150 --cap-factual 0"; D=data ;;
  factual) C="--cap-neutral 0 --cap-creative 0 --cap-factual 500"; D=data ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
JOB=${ARM}_s${START}_c${COUNT}
OUT=output/n512/$JOB
LOG=output/logs/n512_$JOB.log
MARK=~/v/logs/n512_$JOB
if [ -e "$MARK.done" ]; then echo "[n512:$JOB] already done" >> "$LOG"; exit 0; fi
rm -f "$MARK.fail"
echo "[n512:$JOB] START $(date '+%F %T') gpu=$GPU" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt "$COUNT" --trajectory-start "$START" \
  $C --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --data-dir $D --max-new-tokens 200 --batch-size 64 \
  --output-dir "$OUT" >> "$LOG" 2>&1
RC=$?
echo "[n512:$JOB] DONE rc=$RC $(date '+%F %T')" >> "$LOG"
[ $RC -eq 0 ] && touch "$MARK.done" || touch "$MARK.fail"
exit $RC
