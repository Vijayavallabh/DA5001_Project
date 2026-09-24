#!/usr/bin/env bash
# feat-188 (results/onset_prediction_headline_replication.md): every sampled arm of the headline
# re-drawn on --seeds 82 83 84, host B. Two queues, one card each (caution (x)):
#   a  pool neutral, then metered k=10, anchor k=0, opponent k=-1; then judge R2 once rewards exist
#   b  pool creative, pool factual; then merge, rewards, judge R1
# Waits are ORs over .done/.fail with a deadline (caution (c), ninth incident).
# Usage (host B): setsid nohup bash scripts/run_replic.sh a 3 > /dev/null 2>&1 < /dev/null &
set -u
Q=${1:?a|b}; GPU=${2:?gpu}
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
M=${REPLIC_MARKS:-$HOME/v/logs}; R=output/replic
mkdir -p output/logs $R $M
S="--seeds 82 83 84 --max-new-tokens 200 --cap-val 0 --cap-test 0 --cap-attack-train 0 --trust-remote-code"
ALL="--cap-neutral 200 --cap-creative 150 --cap-factual 150"
job() {  # job <name> <cmd...>
  local n=$1; shift; rm -f $M/replic_$n.done $M/replic_$n.fail
  echo "[replic:$n] start $(date '+%F %T') gpu=$GPU" >> output/logs/replic_$n.log
  if "$@" >> output/logs/replic_$n.log 2>&1; then touch $M/replic_$n.done; else touch $M/replic_$n.fail; fi
  echo "[replic:$n] end $(date '+%F %T')" >> output/logs/replic_$n.log
}
waitfor() {  # waitfor <name> ; 0 if .done, 1 if .fail or deadline (8 h)
  local dl=$(( $(date +%s) + 28800 ))
  until [ -e $M/replic_$1.done ] || [ -e $M/replic_$1.fail ]; do
    [ "$(date +%s)" -ge "$dl" ] && return 1; sleep 60; done
  [ -e $M/replic_$1.done ]
}
H1=".venv/bin/python h1.py"
H2H=".venv/bin/python analysis/order_averaged_h2h.py --deecho --seed 7717 --n 64 --k 10 --out results \
  --sel-dir $R/sel_anchor64 --rewards results/selection_rewards64_replic.csv \
  --metered-dir $R/conc_k10 --anchor-dir $R/anchor_k0"
case $Q in
  a)
    job pool_neutral $H1 --k-values 0.0 --trajectories-per-prompt 64 --cap-neutral 200 --cap-creative 0 --cap-factual 0 --batch-size 64 $S --output-dir $R/pool_neutral
    job met_k10  $H1 --k-values 10  --trajectories-per-prompt 1 $ALL --batch-size 48 $S --output-dir $R/conc_k10
    job anchor_k0 $H1 --k-values 0.0 --trajectories-per-prompt 1 $ALL --batch-size 48 $S --output-dir $R/anchor_k0
    job opp      $H1 --k-values -1  --trajectories-per-prompt 1 $ALL --batch-size 48 $S --output-dir $R/opp
    [ -n "${REPLIC_GEN_ONLY:-}" ] && exit 0   # generation moved to another host; judges run where the reward is
    waitfor reward && job judge_R2 $H2H --baseline-dir $R/opp --tag replic_opp ;;
  b)
    job pool_creative $H1 --k-values 0.0 --trajectories-per-prompt 64 --cap-neutral 0 --cap-creative 150 --cap-factual 0 --batch-size 64 $S --output-dir $R/pool_creative
    job pool_factual  $H1 --k-values 0.0 --trajectories-per-prompt 64 --cap-neutral 0 --cap-creative 0 --cap-factual 150 --batch-size 64 $S --output-dir $R/pool_factual
    waitfor pool_neutral || { touch $M/replic_reward.fail; exit 1; }
    mkdir -p $R/sel_anchor64
    for c in neutral creative factual; do ln -sfn "$PWD/$R/pool_$c/trajectories_k0_$c.jsonl" $R/sel_anchor64/; done
    job reward .venv/bin/python analysis/selection_scaling.py --gen-dir $R/sel_anchor64 \
      --baseline-dir output/sweep_plain --reward-cache results/selection_rewards64_replic.csv \
      --rewards-only --batch-size 8 --out results
    waitfor opp && job judge_R1 $H2H --baseline-dir output/sweep_plain --tag replic ;;
esac
echo "[replic:$Q] queue finished $(date '+%F %T')" >> output/logs/replic_queue_$Q.log
