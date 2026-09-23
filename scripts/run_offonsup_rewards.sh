#!/usr/bin/env bash
# feat-172 stage 2 (results/onset_prediction_offsupport_ladder.md): the reward pass over the n=256
# pools, and NOTHING judged. --rewards-only writes the cache and stops, so the registered gate ---
# ranks 0-63 bit-identical to the committed n=64 cache --- is read before any judged number exists.
#
# --batch-size 16 is the committed caches' own (scripts/run_mixpow_rewards.sh for Arm A's reference,
# scripts/run_wscope_score.sh for Arm B's): 256 and 64 are both multiples of it and the reward items
# are prompt-major, so ranks 0-63 of every prompt fall in the same batch tuples. Host B, where both
# references were scored: a bit-identity gate is a constraint on the host (the Arm B lesson in the
# registration's scoring log).
#
# Arm B's draws are sharded (sel_small holds neutral+creative, sel_factual holds factual), so
# sel_merged LINKS the three non-empty class files only; linking every file would clobber real
# shards with the empty ones each shard also writes (scripts/run_wscope_score.sh).
# Usage: run_offonsup_rewards.sh <a|b> <gpu>
set -u
ARM=${1:?a|b}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
mkdir -p output/logs
set -a; . ./.env; set +a
case "$ARM" in
  a) SEL=output/offsup/sel_anchor256; BASE=output/mixpow/baseline; TAG=offsup ;;
  b) SEL=output/onsup/sel_merged; BASE=output/wscope/a_baseline; TAG=onsup
     mkdir -p "$SEL"
     for c in neutral creative; do
       ln -sfn "$PWD/output/onsup/sel_small/trajectories_k0_$c.jsonl" "$SEL/trajectories_k0_$c.jsonl"
     done
     ln -sfn "$PWD/output/onsup/sel_factual/trajectories_k0_factual.jsonl" \
       "$SEL/trajectories_k0_factual.jsonl" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
CACHE=results/${TAG}_rewards256.csv
LOG=output/logs/offonsup_rewards_$ARM.log
MARK="$HOME/v/logs/offonsup_rewards_$ARM"
rm -f "$MARK.done" "$MARK.fail"
if [ -e "$CACHE" ]; then echo "[ofr:$ARM] $CACHE exists; refusing to overwrite (caution (ax))" >> "$LOG"; exit 3; fi
echo "[ofr:$ARM] START $(date +%H:%M:%S) gpu=$GPU gen=$SEL" >> "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_scaling.py \
  --gen-dir "$SEL" --baseline-dir "$BASE" \
  --reward-cache "$CACHE" --tag "_$TAG" --max-n 256 \
  --judges microsoft/Phi-3.5-mini-instruct --batch-size 16 --rewards-only \
  --out results >> "$LOG" 2>&1
RC=$?
echo "[ofr:$ARM] DONE rc=$RC $(date +%H:%M:%S)" >> "$LOG"
if [ $RC -eq 0 ] && [ -s "$CACHE" ]; then touch "$MARK.done"; else touch "$MARK.fail"; fi
