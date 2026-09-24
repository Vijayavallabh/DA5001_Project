#!/usr/bin/env bash
# feat-191 for feat-187's arms (results/he_metrics_note.md: "The AnchoredByte arms of feat-187 are
# added under their own --tag when generated"): Prometheus, then FActScore, on AnchoredByte at each k,
# on one local card, as each k's trajectories are finished on host B (OR wait over .done/.fail there,
# with a deadline). Usage: setsid nohup bash scripts/run_he_ab.sh <gpu> [k ...] > /dev/null 2>&1 < /dev/null &
set -u
GPU=${1:?gpu}; shift; KS="${*:-0.5 0.1 2}"
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
D=output/anchoredbyte/comma7b_70b; M=output/logs/marks; mkdir -p $M $D output/he_metrics
HE=".venv/bin/python analysis/he_metrics.py"
for K in $KS; do
  t=_ab$(echo "$K" | tr -d .)
  dl=$(( $(date +%s) + 21600 ))
  until ssh ${HOSTB:?set HOSTB to the second host} "test -e ~/v/logs/ab70_k$K.done -o -e ~/v/logs/ab70_k$K.fail" < /dev/null; do
    [ "$(date +%s)" -ge "$dl" ] && exit 2; sleep 60; done
  ssh ${HOSTB:?set HOSTB to the second host} "test -e ~/v/logs/ab70_k$K.done" < /dev/null || { touch $M/he$t.fail; continue; }
  rsync -a "${HOSTB:?set HOSTB to the second host}:v/$(basename "$PWD")/$D/trajectories_k${K}_*.jsonl" $D/ < /dev/null || { touch $M/he$t.fail; continue; }
  { $HE --do-dump --dump output/he_metrics/arms$t.jsonl --arm "ab_k$K=traj:$D:$K" &&
    $HE --prometheus --dump output/he_metrics/arms$t.jsonl --tag "$t" --out results &&
    $HE --factscore --dump output/he_metrics/arms$t.jsonl --tag "$t" --out results; } \
    < /dev/null >> output/logs/he$t.log 2>&1 && touch $M/he$t.done || touch $M/he$t.fail
done
