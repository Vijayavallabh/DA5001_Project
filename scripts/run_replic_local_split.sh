#!/usr/bin/env bash
# feat-188, local: queue a's three single-draw arms moved to a second card while the neutral pool
# finishes on the first (its queue shell was killed; the pool's h1.py child keeps running, caution (c)).
# Then R0, then hand the four directories to host B and release its waiting queue.
# Usage: setsid nohup bash scripts/run_replic_local_split.sh <gpu> <pool-pid> > /dev/null 2>&1 < /dev/null &
set -u
GPU=${1:?gpu}; POOL=${2:?pid of the neutral pool h1.py}
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
M=output/logs/marks; R=output/replic; mkdir -p $M
S="--seeds 82 83 84 --max-new-tokens 200 --cap-val 0 --cap-test 0 --cap-attack-train 0 --trust-remote-code"
ALL="--cap-neutral 200 --cap-creative 150 --cap-factual 150"
job() {  # as in run_replic.sh
  local n=$1; shift; rm -f $M/replic_$n.done $M/replic_$n.fail
  echo "[replic:$n] start $(date '+%F %T') gpu=$GPU" >> output/logs/replic_$n.log
  if "$@" >> output/logs/replic_$n.log 2>&1; then touch $M/replic_$n.done; else touch $M/replic_$n.fail; fi
  echo "[replic:$n] end $(date '+%F %T')" >> output/logs/replic_$n.log
}
H1=".venv/bin/python h1.py"
job met_k10  $H1 --k-values 10  --trajectories-per-prompt 1 $ALL --batch-size 48 $S --output-dir $R/conc_k10
job anchor_k0 $H1 --k-values 0.0 --trajectories-per-prompt 1 $ALL --batch-size 48 $S --output-dir $R/anchor_k0
job opp      $H1 --k-values -1  --trajectories-per-prompt 1 $ALL --batch-size 48 $S --output-dir $R/opp
rm -f $M/replic_R0.done $M/replic_R0.fail
.venv/bin/python analysis/order_averaged_h2h.py --deecho --seed 7717 --n 64 --k 10 --out results \
  --sel-dir output/phase5/sel_anchor64_seed52 --rewards results/selection_rewards64_seed52.csv \
  --metered-dir output/phase2/conc_all --anchor-dir output/sweep_plain --baseline-dir output/sweep_plain \
  --tag seed52_deecho > output/logs/replic_R0.log 2>&1 && touch $M/replic_R0.done || touch $M/replic_R0.fail
# the pool: wait on its PID (never on a pattern), deadline 3 h, then check it is whole
dl=$(( $(date +%s) + 10800 ))
while kill -0 "$POOL" 2>/dev/null && [ "$(date +%s)" -lt "$dl" ]; do sleep 30; done
F=$R/pool_neutral/trajectories_k0_neutral.jsonl
if [ "$(wc -l < $F)" -eq 12800 ] && ! kill -0 "$POOL" 2>/dev/null; then touch $M/replic_pool_neutral.done
else touch $M/replic_pool_neutral.fail; exit 1; fi
for n in met_k10 anchor_k0 opp; do [ -e $M/replic_$n.done ] || exit 1; done
rsync -a $R/pool_neutral $R/conc_k10 $R/anchor_k0 $R/opp PrakashDGX_H2:v/DA5001_Project/$R/ \
  > output/logs/replic_sync.log 2>&1 || exit 1
ssh PrakashDGX_H2 'touch ~/v/logs/replic_pool_neutral.done ~/v/logs/replic_opp.done' \
  >> output/logs/replic_sync.log 2>&1 && touch $M/replic_synced.done
