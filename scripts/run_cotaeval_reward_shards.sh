#!/usr/bin/env bash
# feat-193: the pool's 32,000 rewards over four cards, prompt-sharded (selection_scaling.py --shard;
# every batch is 8 candidates of one prompt, so the shards build the single run's batches), merged in
# the single run's row order, then the registered scorer. Usage: run_cotaeval_reward_shards.sh 0 1 2 4
set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
set -a; . ./.env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache
M=output/logs/marks; C=results/selection_rewards64_cotaeval_inf.csv; N=$#
[ -e "$C" ] && { echo "$C exists; refusing to overwrite" >> output/logs/cta_reward.log; exit 1; }
i=0
for g in "$@"; do
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python analysis/selection_scaling.py --gen-dir output/cotaeval_inf/pool \
    --baseline-dir output/cotaeval_inf/arms --reward-cache $C --rewards-only --batch-size 8 \
    --shard $i/$N --out results > output/logs/cta_reward_shard$i.log 2>&1 < /dev/null &
  i=$((i+1))
done
wait
for i in $(seq 0 $((N-1))); do [ -s "$C.shard${i}of$N" ] || { touch $M/cta_reward.fail; exit 1; }; done
.venv/bin/python - "$C" "$N" <<'PY'
import csv, sys
c, n = sys.argv[1], int(sys.argv[2])
rows = []
for i in range(n):
    r = list(csv.reader(open(f"{c}.shard{i}of{n}")))
    head, body = r[0], r[1:]
    rows += body
rows.sort(key=lambda r: (r[0], int(r[1])))
assert len(rows) == 32000 and len({(r[0], r[1]) for r in rows}) == 32000, len(rows)
with open(c, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(head)
    w.writerows(rows)
print(f"merged {len(rows)} rows into {c}")
PY
[ -s "$C" ] && touch $M/cta_reward.done || { touch $M/cta_reward.fail; exit 1; }
.venv/bin/python analysis/cotaeval_infringement.py --out results > output/logs/cta_score.log 2>&1 \
  && touch $M/cta_score.done || touch $M/cta_score.fail
