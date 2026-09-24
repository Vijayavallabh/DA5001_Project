#!/usr/bin/env bash
# feat-189 (results/onset_prediction_hybrid.md): drafts from the pathwise meter at B = C - log n,
# C = log 64, then the committed reward over each pool, then one judged levels pass.
# Usage: run_hybrid.sh <gpu>      (one queue shell on one card, caution (x))
set -u
GPU=${1:?gpu}
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
M=${HYBRID_MARKS:-output/logs/marks}; D=output/hybrid
mkdir -p output/logs $D $M
C="--constraint pathwise --no-prefix-debt --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
   --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 --batch-size 64 --trust-remote-code"
job() { local n=$1; shift; rm -f $M/hybrid_$n.done $M/hybrid_$n.fail
  echo "[hybrid:$n] start $(date '+%F %T') gpu=$GPU" >> output/logs/hybrid_$n.log
  if "$@" >> output/logs/hybrid_$n.log 2>&1; then touch $M/hybrid_$n.done; else touch $M/hybrid_$n.fail; return 1; fi
  echo "[hybrid:$n] end $(date '+%F %T')" >> output/logs/hybrid_$n.log; }
H1=".venv/bin/python h1.py"
RW=".venv/bin/python analysis/selection_scaling.py --rewards-only --batch-size 8 --baseline-dir output/sweep_plain --out results"
job n8 $H1 --k-values 0.010397 --trajectories-per-prompt 8 $C --output-dir $D/pw_n8
job n2 $H1 --k-values 0.017329 --trajectories-per-prompt 2 $C --output-dir $D/pw_n2
job n1 $H1 --k-values 0.020794 --trajectories-per-prompt 1 $C --output-dir $D/pw_n1
job rw8 $RW --gen-dir $D/pw_n8 --k-token 0.010397 --max-n 8 --reward-cache results/selection_rewards8_hybrid.csv
job rw2 $RW --gen-dir $D/pw_n2 --k-token 0.017329 --max-n 2 --reward-cache results/selection_rewards2_hybrid.csv
P=output/phase5/sel_anchor64; R=results/selection_rewards64.csv
job judge .venv/bin/python analysis/levels_pass.py --tag hybrid \
  --arm sel64=sel:$P:$R:64 --arm sel8=sel:$P:$R:8 --arm sel1=sel:$P:$R:1 \
  --arm hyb8=sel:$D/pw_n8:results/selection_rewards8_hybrid.csv:8:0.010397 \
  --arm hyb2=sel:$D/pw_n2:results/selection_rewards2_hybrid.csv:2:0.017329 \
  --arm pw1=traj:$D/pw_n1:0.020794:pathwise
echo "[hybrid] queue finished $(date '+%F %T')" >> output/logs/hybrid_queue.log
