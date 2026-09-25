#!/usr/bin/env bash
# feat-213 (results/onset_prediction_long_outputs.md): every mechanism at T_max = 1000, host B GPUs 0-3.
# One queue shell per card (caution (x)); a cross-card wait is an OR over .done/.fail per job, with a
# deadline (caution (c)), and a .fail or the deadline stops the queue.   bash scripts/run_feat213.sh q0|q1|q2|q3
set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
L=~/v/logs; O=output/feat213; mkdir -p "$L" output/logs $O
run() {  # run <job> <cmd...>
  local j=$1; shift
  rm -f "$L/f213_$j.done" "$L/f213_$j.fail"
  echo "[f213:$j] START $(date '+%F %T')"
  if "$@" > "output/logs/f213_$j.log" 2>&1; then touch "$L/f213_$j.done"; echo "[f213:$j] ok $(date '+%F %T')"; return 0
  else touch "$L/f213_$j.fail"; echo "[f213:$j] FAIL $(date '+%F %T')"; return 1; fi
}
wait_for() {  # wait_for <deadline-seconds> <job>...
  local dl=$(( $(date +%s) + $1 )); shift
  for j in "$@"; do
    until [ -e "$L/f213_$j.done" ] || [ -e "$L/f213_$j.fail" ]; do
      [ "$(date +%s)" -gt "$dl" ] && { echo "[f213] deadline waiting for $j"; return 1; }
      sleep 30
    done
    [ -e "$L/f213_$j.done" ] || { echo "[f213] $j failed"; return 1; }
  done
}
H1=(h1.py --trajectories-per-prompt 1 --seeds 61 62 63 --cap-neutral 200 --cap-factual 150 --cap-creative 150
    --cap-val 0 --cap-test 0 --cap-attack-train 0 --batch-size 48 --trust-remote-code --max-new-tokens 1000)
ARMS=(--arm sel_n64=arm:$O/sel:whole1000n64:kl --arm sel_n16=arm:$O/sel:whole1000n16:kl
      --arm sel_n4=arm:$O/sel:whole1000n4:kl --arm sel_n1=arm:$O/sel:whole1000n1:kl
      --arm inst=arm:$O/inst:blk100n8:kl --arm anchor_k0=arm:$O/kl:0:kl
      --arm kl_20.8=arm:$O/kl:0.020794:kl --arm kl_0.1=arm:$O/kl:0.1:kl --arm kl_0.5=arm:$O/kl:0.5:kl
      --arm kl_10=arm:$O/kl:10:kl --arm pw_4.16=arm:$O/pw:0.0041589:pathwise
      --arm pw_20.8=arm:$O/pw:0.020794:pathwise --arm front_4.16=arm:$O/front:1e-09:pathwise
      --arm win_4.16=arm:$O/win:4.1589:pathwise
      --control sel_n64=sel_n1 --control sel_n16=sel_n1 --control sel_n4=sel_n1 --control inst=sel_n1
      --control kl_20.8=anchor_k0 --control kl_0.1=anchor_k0 --control kl_0.5=anchor_k0 --control kl_10=anchor_k0
      --control pw_4.16=anchor_k0 --control pw_20.8=anchor_k0 --control front_4.16=anchor_k0
      --control win_4.16=anchor_k0
      --diff inst,pw_20.8 --diff inst,kl_20.8 --diff sel_n64,pw_4.16 --diff sel_n64,front_4.16
      --diff sel_n64,win_4.16 --diff sel_n64,kl_0.1 --diff sel_n64,kl_0.5 --diff inst,sel_n64 --diff sel_n64,kl_10)
MH=($PY analysis/matched_h2h.py --baseline-dir $O/kl --judge-max-chars 0 --fit-window 8180)
case "${1:?q0|q1|q2|q3}" in
  q0) export CUDA_VISIBLE_DEVICES=0
      run sel $PY analysis/blockwise_selection.py --block-len 1000 --t-max 1000 --n 64 --scorer reward \
        --reward-max-chars 0 --pool-arms 1 4 16 64 --out-dir $O/sel || exit 1
      wait_for 21600 inst kl meters || exit 1
      export CUDA_VISIBLE_DEVICES=0,1
      run judge_G "${MH[@]}" --tag f213_G --judge google/gemma-2-27b-it --device-map auto "${ARMS[@]}" ;;
  # 2026-09-25: q0's pool was stopped before writing output. At 256 rows its anchor KV cache alone reaches
  # ~58 GB at 1,150 tokens beside the 7B scorer, and the batches are sorted by ascending prompt length, so
  # the last ones would not fit (q1 died of exactly this). Re-run from scratch at half the draw batch.
  q0b) export CUDA_VISIBLE_DEVICES=0
      run sel $PY analysis/blockwise_selection.py --block-len 1000 --t-max 1000 --n 64 --scorer reward \
        --reward-max-chars 0 --pool-arms 1 4 16 64 --gen-batch 128 --out-dir $O/sel || exit 1
      wait_for 21600 inst kl meters || exit 1
      export CUDA_VISIBLE_DEVICES=0,1
      run judge_G "${MH[@]}" --tag f213_G --judge google/gemma-2-27b-it --device-map auto "${ARMS[@]}" ;;
  q1) export CUDA_VISIBLE_DEVICES=1
      run inst $PY analysis/blockwise_selection.py --block-len 100 --t-max 1000 --n 8 --scorer value \
        --reward-max-chars 0 --out-dir $O/inst ;;
  # 2026-09-25: q1 ran out of memory at block 5 of 10 (anchor rollouts of 256 rows at ~1,100-token contexts
  # beside the resident 7B scorer), before writing any output; re-run from scratch at half the draw batch.
  q1b) export CUDA_VISIBLE_DEVICES=1
      run inst $PY analysis/blockwise_selection.py --block-len 100 --t-max 1000 --n 8 --scorer value \
        --reward-max-chars 0 --gen-batch 128 --out-dir $O/inst ;;
  q2) export CUDA_VISIBLE_DEVICES=2
      run kl $PY "${H1[@]}" --k-values -1 0 0.020794 0.1 0.5 10 --output-dir $O/kl || exit 1
      wait_for 21600 sel inst meters || exit 1
      run judge_B "${MH[@]}" --tag f213_B --judge microsoft/Phi-3.5-mini-instruct "${ARMS[@]}" ;;
  q3) export CUDA_VISIBLE_DEVICES=3
      run meters bash -c "$PY ${H1[*]} --constraint pathwise --no-prefix-debt --k-values 0.0041589 0.020794 --output-dir $O/pw && \
        $PY ${H1[*]} --constraint pathwise --no-prefix-debt --initial-bank 4.158883 --k-values 1e-9 --output-dir $O/front && \
        $PY ${H1[*]} --constraint pathwise --window 50 --no-prefix-debt --k-values 4.1589 --output-dir $O/win" ;;
  *) echo "unknown queue $1"; exit 2 ;;
esac
