#!/usr/bin/env bash
# feat-211 (results/onset_prediction_cpk_baseline.md) and feat-212 (results/onset_prediction_scorer_judge_factorial.md),
# host B GPUs 4-7. One queue shell per card (caution (x)); a cross-card wait is an OR over .done/.fail for each job,
# with a deadline (caution (c)), and a .fail or the deadline stops the queue rather than running on stale inputs.
#   bash scripts/run_feat211.sh q4|q5|q6|q7|q67
set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
L=~/v/logs; mkdir -p "$L" output/logs output/feat211
F=output/feat210; O=output/feat211
run() {  # run <job> <cmd...>
  local j=$1; shift
  rm -f "$L/f211_$j.done" "$L/f211_$j.fail"
  echo "[f211:$j] START $(date '+%F %T')"
  if "$@" > "output/logs/f211_$j.log" 2>&1; then touch "$L/f211_$j.done"; echo "[f211:$j] ok $(date '+%F %T')"; return 0
  else touch "$L/f211_$j.fail"; echo "[f211:$j] FAIL $(date '+%F %T')"; return 1; fi
}
wait_for() {  # wait_for <deadline-seconds> <job>... : every job .done; a .fail or the deadline aborts
  local dl=$(( $(date +%s) + $1 )); shift
  for j in "$@"; do
    until [ -e "$L/f211_$j.done" ] || [ -e "$L/f211_$j.fail" ]; do
      [ "$(date +%s)" -gt "$dl" ] && { echo "[f211] deadline waiting for $j"; return 1; }
      sleep 30
    done
    [ -e "$L/f211_$j.done" ] || { echo "[f211] $j failed"; return 1; }
  done
}
gen() {  # gen <job> <base-seed> <dir>: 16 risky draws per prompt at k=-1, per-step log carries both probabilities
  run "$1" $PY h1.py --k-values -1 --trajectories-per-prompt 16 --seeds "$2" --cap-neutral 200 --cap-factual 150 \
    --cap-creative 150 --cap-val 0 --cap-test 0 --cap-attack-train 0 --batch-size 48 --trust-remote-code --output-dir "$3"
}
# feat-212's five arms, in every pass
ARMS=(--arm sel_n64=sel:64 --arm sel_n1=sel:1 --arm sel_g64=sel:64:results/selection_rewards64_gemma27b.csv
      --arm anchor_k0=arm:output/sweep_plain:0:kl --arm metered_k10=arm:output/phase2/conc_all:10:kl
      --control sel_n64=sel_n1 --control sel_g64=sel_n1 --control metered_k10=anchor_k0
      --diff sel_n64,sel_g64 --diff sel_n64,metered_k10 --diff sel_g64,metered_k10)
# feat-211's arms and the matched-certificate comparators, in the judge-B and judge-G passes
CPK=(--arm opp_r1=base:1 --arm blk10n64=arm:output/feat201/blk10n64:blk10n64:kl
     --arm blk25n64=arm:output/feat201/blk25n64:blk25n64:kl --arm blk200n1=arm:output/feat201/blk200n1:blk200n1:kl
     --arm win_0=arm:$F/win_plain:0:pathwise --arm win_4.16=arm:$F/win_plain:4.1589:pathwise
     --arm frontpw_4.16=arm:$F/front_plain:1e-09:pathwise --arm pw_83.18=arm:$F/pw_plain:0.415888:pathwise
     --arm pw_33.27=arm:$F/pw_plain:0.166355:pathwise
     --arm cpk_4.16=arm:$O/cpk_arms:cpk4.1589:kl --arm cpk_33.27=arm:$O/cpk_arms:cpk33.271:kl
     --arm cpk_83.18=arm:$O/cpk_arms:cpk83.178:kl --arm cpk_159.83=arm:$O/cpk_arms:cpk159.83:kl
     --control opp_r1=anchor_k0 --control blk10n64=blk200n1 --control blk25n64=blk200n1
     --control win_4.16=win_0 --control frontpw_4.16=win_0 --control pw_83.18=win_0 --control pw_33.27=win_0
     --control cpk_4.16=anchor_k0 --control cpk_33.27=anchor_k0 --control cpk_83.18=anchor_k0
     --control cpk_159.83=anchor_k0
     --diff blk10n64,cpk_83.18 --diff blk25n64,cpk_33.27 --diff sel_n64,cpk_4.16
     --diff sel_g64,win_4.16 --diff sel_g64,frontpw_4.16
     --diff cpk_83.18,pw_83.18 --diff cpk_33.27,pw_33.27 --diff cpk_159.83,metered_k10)
MH=($PY analysis/matched_h2h.py)
case "${1:?q4|q5|q6|q7|q67}" in
  q4) export CUDA_VISIBLE_DEVICES=4
      gen gen_a 1101 $O/risky_a
      wait_for 10800 gen_a gen_b gen_c gen_d || exit 1
      run arms $PY analysis/cpk_baseline.py --risky-dirs $O/risky_a $O/risky_b $O/risky_c $O/risky_d || exit 1
      run judge_B "${MH[@]}" --tag cpk_B_hostb --judge microsoft/Phi-3.5-mini-instruct "${ARMS[@]}" "${CPK[@]}"
      run fact_F "${MH[@]}" --tag fact_F --judge Qwen/Qwen2.5-14B-Instruct --device-map auto "${ARMS[@]}" ;;
  q5) export CUDA_VISIBLE_DEVICES=5
      gen gen_b 1102 $O/risky_b
      wait_for 14400 arms || exit 1
      # the committed judge-G verdicts were judged on THIS host, so they seed this pass's cache
      [ -e results/matched_h2h_verdicts_cpk_G.csv ] || \
        cp results/matched_h2h_verdicts_matched_plain_G.csv results/matched_h2h_verdicts_cpk_G.csv
      run judge_G "${MH[@]}" --tag cpk_G --judge google/gemma-2-27b-it --device-map auto "${ARMS[@]}" "${CPK[@]}" ;;
  q6) export CUDA_VISIBLE_DEVICES=6
      gen gen_c 1103 $O/risky_c
      run leak $PY analysis/cpk_extraction.py --risky-model output/memorizing_llama8b --out results ;;
  q7) export CUDA_VISIBLE_DEVICES=7
      gen gen_d 1104 $O/risky_d
      run fact_C "${MH[@]}" --tag fact_C --judge meta-llama/Meta-Llama-3.1-8B-Instruct "${ARMS[@]}" ;;
  # 2026-09-25: q4 and q5 were stopped before any verdict was read, because the first `arms` summed R only
  # through an <|eot_id|> the plain harness does not stop at (analysis/cpk_baseline.py served_steps). These
  # re-enter them after the draws: arms again, then the same passes (the non-CP-k verdicts are cached).
  q4b) export CUDA_VISIBLE_DEVICES=4
      run arms $PY analysis/cpk_baseline.py --risky-dirs $O/risky_a $O/risky_b $O/risky_c $O/risky_d || exit 1
      run judge_B "${MH[@]}" --tag cpk_B_hostb --judge microsoft/Phi-3.5-mini-instruct "${ARMS[@]}" "${CPK[@]}"
      run fact_F "${MH[@]}" --tag fact_F --judge Qwen/Qwen2.5-14B-Instruct --device-map auto "${ARMS[@]}" ;;
  q5b) export CUDA_VISIBLE_DEVICES=5
      sleep 60; wait_for 3600 arms || exit 1
      run judge_G "${MH[@]}" --tag cpk_G --judge google/gemma-2-27b-it --device-map auto "${ARMS[@]}" "${CPK[@]}" ;;
  q67) wait_for 21600 leak fact_C || exit 1
      export CUDA_VISIBLE_DEVICES=6,7
      run fact_D "${MH[@]}" --tag fact_D --judge Qwen/Qwen2.5-72B-Instruct --device-map auto "${ARMS[@]}"
      run fact_E "${MH[@]}" --tag fact_E --judge mistralai/Mixtral-8x7B-Instruct-v0.1 --device-map auto "${ARMS[@]}" ;;
  *) echo "unknown queue $1"; exit 2 ;;
esac
