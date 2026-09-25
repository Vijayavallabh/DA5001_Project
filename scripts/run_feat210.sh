#!/usr/bin/env bash
# feat-210 (results/onset_prediction_windowed_meter.md): the sliding-window meter and the matched
# per-token pathwise meters, generation and leakage on host B. One queue shell per card, jobs in order
# (caution (x)); sentinels in output/logs.   bash scripts/run_feat210.sh q4 | q5 | q6 | q7
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
L=output/logs
O=output/feat210
mkdir -p $L $O
run() {  # run <name> <cmd...>
  local name=$1; shift
  echo "[$name] start $(date '+%F %T')"
  if "$@"; then touch "$L/$name.done"; echo "[$name] exit=0 $(date '+%F %T')"
  else local rc=$?; touch "$L/$name.fail"; echo "[$name] exit=$rc $(date '+%F %T')"; fi
}
H1="h1.py --trajectories-per-prompt 1 --seeds 52 53 54 --cap-neutral 200 --cap-factual 150 --cap-creative 150
    --cap-val 0 --cap-test 0 --cap-attack-train 0 --batch-size 48 --trust-remote-code"
WIN="--constraint pathwise --window 50 --no-prefix-debt"
CA="analysis/composition_attack.py --risky-model output/memorizing_llama8b --split attack_train --limit 100
    --modes single --constraint pathwise --window 50 --no-prefix-debt"
case "${1:?queue}" in
  q4) export CUDA_VISIBLE_DEVICES=${GPU:-4}
      run feat210_win_plain $PY $H1 $WIN --k-values 0 4.1589 12.4767 24.9534 40 125 --output-dir $O/win_plain ;;
  q5) export CUDA_VISIBLE_DEVICES=${GPU:-5}
      run feat210_win_chat $PY $H1 $WIN --use-chat-template --k-values 0 12.4767 24.9534 40 125 --output-dir $O/win_chat ;;
  q6) export CUDA_VISIBLE_DEVICES=${GPU:-6}
      run feat210_pw_plain $PY $H1 --constraint pathwise --no-prefix-debt --k-values 0.415888 0.166355 --output-dir $O/pw_plain
      run feat210_front_plain $PY $H1 --constraint pathwise --no-prefix-debt --initial-bank 4.158883 --k-values 1e-9 --output-dir $O/front_plain ;;
  q7) export CUDA_VISIBLE_DEVICES=${GPU:-7}
      mkdir -p $O/leak_plain $O/leak_chat
      run feat210_leak_plain $PY $CA --k-values -1 0 4.1589 12.4767 24.9534 40 60 80 100 125 150 200 \
        --out $O/leak_plain --figures $O/leak_plain --text-out $O/leak_plain/extracted.csv --queries-out $O/leak_plain/queries.jsonl
      run feat210_leak_chat $PY $CA --use-chat-template --k-values -1 0 24.9534 40 80 125 \
        --out $O/leak_chat --figures $O/leak_chat --text-out $O/leak_chat/extracted.csv --queries-out $O/leak_chat/queries.jsonl ;;
esac
