#!/usr/bin/env bash
# feat-182 (results/onset_prediction_selector_redraw.md): feat-179 Part A re-drawn with --seed 5678,
# every other flag unchanged, on the LOCAL A100s where Part A ran (so the seed is the only change).
# One queue per card (caution (x)); each waits for that card's last feat-179/180 job by the line
# that job's own launcher writes (caution (c): a filesystem condition, OR'd where there are two
# ways to end, with a deadline).
# Usage: setsid nohup bash scripts/run_selfix_redraw_local.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh
set -a; . ./.env; set +a
MEM=output/memorizing_llama8b
Q=output/logs/selfix_redraw_queue.log
END=$(( $(date +%s) + 16 * 3600 ))
# waitany <log> <pattern> [<log> <pattern>]: true once ANY pair matches; false at the deadline
waitany() {
  while :; do
    local a=("$@")
    for ((i = 0; i < ${#a[@]}; i += 2)); do grep -qE "${a[i+1]}" "${a[i]}" 2>/dev/null && return 0; done
    [ "$(date +%s)" -ge "$END" ] && { echo "[redraw] deadline waiting on $1" >> "$Q"; return 1; }
    sleep 60
  done
}
red() {   # red <gpu> <tag> <anchor dir under output/phase5> [experts impl]
  local G=$1 P=selfixR_$2 LOG=output/logs/selfixR_$2.log
  if [ -s "results/${P}_per_passage.csv" ]; then echo "[$P] already on disk" >> "$LOG"; return; fi
  echo "[$P] start $(date '+%F %T') on GPU $G" >> "$LOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_extraction.py --safe-model "output/phase5/$3" --risky-model $MEM \
    --n-values 1 8 64 256 --limit 100 --batch-size 32 --seed 5678 ${4:+--experts-impl "$4"} \
    --prefix "$P" --out results >> "$LOG" 2>&1
  echo "[$P] exit=$? at $(date '+%F %T')" >> "$LOG"
}
g1() { waitany output/logs/vetladder_L150_pleias3b.log '^\[vetladder_L150_pleias3b\] (exit=|already on disk)' || return
       red 1 kl3m37b mem_kl3m-003-3_7b eager; red 1 llama32_1b mem_llama32-1b
       red 1 pleias350m mem_Pleias-350m-Preview; red 1 kl3m170m mem_kl3m-002-170m; }
g2() { waitany output/logs/vetladder_L150_comma1t.log '^\[vetladder_L150_comma1t\] (exit=|already on disk)' || return
       red 2 kl3m520m mem_kl3m-002-520m eager; red 2 qwen25_7b mem_qwen25-7b
       red 2 llama32_3b mem_llama32-3b; red 2 opencalm1b mem_opencalm1b; }
g4() { waitany output/logs/selfix_clean_grid64.log '^\[selfix_clean_grid64\] exit=' \
               output/logs/selfix_grid64_waiter.log 'deadline reached|already on disk' || return
       red 4 pleias12b mem_Pleias-1_2b-Preview; red 4 opencalm3b mem_opencalm3b
       red 4 kl3m17b mem_kl3m-003-1_7b; red 4 phi35mini mem_phi35mini; }
echo "[redraw] armed $(date '+%F %T')" >> "$Q"
g1 & g2 & g4 & wait
echo "[redraw] drained $(date '+%F %T')" >> "$Q"
