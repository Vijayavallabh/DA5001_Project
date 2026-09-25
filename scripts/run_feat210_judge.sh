#!/usr/bin/env bash
# feat-210 judging (results/onset_prediction_windowed_meter.md), local A100s: one pass per judge over
# every arm at a matched certificate, verdicts cached per arm (analysis/matched_h2h.py), so the committed
# arms are judged first and the feat-210 arms join the same pass when host B's generations are pulled.
#   bash scripts/run_feat210_judge.sh <plain|chat> <B|G> [existing]
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/gpu_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf_cache"
PY=.venv/bin/python
F=output/feat210
case "$2" in
  B) J=(--judge microsoft/Phi-3.5-mini-instruct) ;;
  G) J=(--judge google/gemma-2-27b-it --device-map auto) ;;
esac
if [ "$1" = plain ]; then
  A=(--arm sel_n64=sel:64 --arm sel_n1=sel:1 --arm metered_k10=arm:output/phase2/conc_all:10:kl
     --arm anchor_k0=arm:output/sweep_plain:0:kl --arm nonempty=arm:output/feat209/pool:nonempty:kl
     --arm blk10n64=arm:output/feat201/blk10n64:blk10n64:kl --arm blk25n64=arm:output/feat201/blk25n64:blk25n64:kl
     --arm blk200n1=arm:output/feat201/blk200n1:blk200n1:kl --arm met_k0.5=arm:output/phase2/conc_all:0.5:kl
     --arm frontkl_4.16=arm:output/phase5/sparse_b4.16_t0:1e-09:kl --arm frontkl_64=arm:output/phase5/sparse_b64_t0:1e-09:kl
     --arm opp_r1=base:1)
  C=(--control sel_n64=sel_n1 --control nonempty=sel_n1 --control blk10n64=blk200n1 --control blk25n64=blk200n1
     --control metered_k10=anchor_k0 --control met_k0.5=anchor_k0 --control frontkl_4.16=anchor_k0
     --control frontkl_64=anchor_k0 --control opp_r1=anchor_k0)
  D=()
  if [ "${3:-}" != existing ]; then
    for w in 4.1589:win_4.16 12.4767:win_12.48 24.9534:win_24.95 40:win_40 125:win_125; do
      A+=(--arm ${w#*:}=arm:$F/win_plain:${w%%:*}:pathwise); C+=(--control ${w#*:}=win_0); done
    A+=(--arm win_0=arm:$F/win_plain:0:pathwise --arm pw_83.18=arm:$F/pw_plain:0.415888:pathwise
        --arm pw_33.27=arm:$F/pw_plain:0.166355:pathwise --arm frontpw_4.16=arm:$F/front_plain:1e-09:pathwise)
    C+=(--control pw_83.18=win_0 --control pw_33.27=win_0 --control frontpw_4.16=win_0)
    D=(--diff sel_n64,win_4.16 --diff sel_n64,frontpw_4.16                                   # P1
       --diff sel_n64,win_24.95 --diff sel_n64,win_40 --diff sel_n64,win_125               # P2
       --diff blk10n64,pw_83.18 --diff blk25n64,pw_33.27                                    # P3
       --diff blk10n64,win_24.95 --diff blk25n64,win_12.48)                                 # P4
  fi
  D+=(--diff sel_n64,metered_k10 --diff blk10n64,met_k0.5 --diff sel_n64,met_k0.5 --diff nonempty,metered_k10
      --diff sel_n64,frontkl_4.16 --diff sel_n64,frontkl_64)                                # descriptive
  $PY analysis/matched_h2h.py --tag matched_plain_$2 "${J[@]}" "${A[@]}" "${C[@]}" "${D[@]}"
else
  A=(--arm sel_n64=sel:64 --arm sel_n1=sel:1 --arm nonempty=arm:output/feat209/pool:nonempty:kl
     --arm chat_k10=arm:output/feat184/chat_k10:10:kl --arm chat_anchor=arm:output/sweep_chat:0:kl)
  C=(--control sel_n64=sel_n1 --control nonempty=sel_n1 --control chat_k10=chat_anchor)
  D=()
  if [ "${3:-}" != existing ]; then
    for w in 12.4767:winc_12.48 24.9534:winc_24.95 40:winc_40 125:winc_125; do
      A+=(--arm ${w#*:}=arm:$F/win_chat:${w%%:*}:pathwise); C+=(--control ${w#*:}=winc_0); done
    A+=(--arm winc_0=arm:$F/win_chat:0:pathwise)
    D=(--diff sel_n64,winc_125 --diff sel_n64,winc_12.48                                   # P5
       --diff sel_n64,winc_24.95 --diff sel_n64,winc_40)
  fi
  D+=(--diff sel_n64,chat_k10 --diff nonempty,chat_k10)
  $PY analysis/matched_h2h.py --tag matched_chat_$2 --baseline-dir output/sweep_chat "${J[@]}" "${A[@]}" "${C[@]}" "${D[@]}"
fi
