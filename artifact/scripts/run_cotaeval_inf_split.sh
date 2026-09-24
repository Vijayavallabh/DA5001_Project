#!/usr/bin/env bash
# feat-193, re-dealt: the 64-draw pool split by trajectory index across two cards. Seeds are
# index-only (dap/stats.py:build_trajectory_seeds) and prompts sharing a seed are batched together,
# so --trajectory-start 0/32 with 32 each makes exactly the draws one 64-draw run makes. The arms
# run was started by run_cotaeval_inf.sh, whose queue shell was killed; its h1.py child keeps
# running (caution (c)) and is waited on by PID, never by pattern.
# Usage: setsid nohup bash scripts/run_cotaeval_inf_split.sh <gpuA> <arms pid> <gpuB> <pid B waits on>
#        > /dev/null 2>&1 < /dev/null &
set -u
GA=${1:?}; ARMS=${2:?}; GB=${3:?}; WB=${4:?}
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
M=output/logs/marks; O=output/cotaeval_inf; mkdir -p $M
A=jacquelinehe/tinycomma-1.8b-llama3-tokenizer
C="--data-dir data/bench/cotaeval_inf --cap-factual 500 --cap-neutral 0 --cap-creative 0 --cap-val 0 \
   --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 --trust-remote-code"
job() { local n=$1; shift; rm -f $M/cta_$n.done $M/cta_$n.fail
  echo "[cta:$n] start $(date '+%F %T') gpu=$CUDA_VISIBLE_DEVICES" >> output/logs/cta_$n.log
  if "$@" >> output/logs/cta_$n.log 2>&1; then touch $M/cta_$n.done; else touch $M/cta_$n.fail; fi
  echo "[cta:$n] end $(date '+%F %T')" >> output/logs/cta_$n.log; }
waitpid() { local dl=$(( $(date +%s) + 14400 ))
  while kill -0 "$1" 2>/dev/null && [ "$(date +%s)" -lt "$dl" ]; do sleep 30; done
  ! kill -0 "$1" 2>/dev/null; }
POOL=".venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 32 --batch-size 64 $C \
  --safe-model-path $A --risky-model-path $A"
( waitpid "$ARMS" || { touch $M/cta_arms.fail; exit 1; }
  for k in -1 0 0.5 1 10; do
    [ "$(cat $O/arms/trajectories_k${k}_factual.jsonl 2>/dev/null | wc -l)" -eq 500 ] || { touch $M/cta_arms.fail; exit 1; }
  done
  touch $M/cta_arms.done
  CUDA_VISIBLE_DEVICES=$GA job pool_a $POOL --trajectory-start 0 --output-dir $O/pool_a ) &
( waitpid "$WB"
  CUDA_VISIBLE_DEVICES=$GB job pool_b $POOL --trajectory-start 32 --output-dir $O/pool_b ) &
wait
[ -e $M/cta_pool_a.done ] && [ -e $M/cta_pool_b.done ] || { touch $M/cta_pool.fail; exit 1; }
mkdir -p $O/pool
cat $O/pool_a/trajectories_k0_factual.jsonl $O/pool_b/trajectories_k0_factual.jsonl \
  > $O/pool/trajectories_k0_factual.jsonl
[ "$(wc -l < $O/pool/trajectories_k0_factual.jsonl)" -eq 32000 ] || { touch $M/cta_pool.fail; exit 1; }
touch $M/cta_pool.done
export CUDA_VISIBLE_DEVICES=$GA
job reward .venv/bin/python analysis/selection_scaling.py --gen-dir $O/pool \
  --baseline-dir $O/arms --reward-cache results/selection_rewards64_cotaeval_inf.csv \
  --rewards-only --batch-size 8 --out results
job score .venv/bin/python analysis/cotaeval_infringement.py --out results
