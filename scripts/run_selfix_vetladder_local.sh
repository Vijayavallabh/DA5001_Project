#!/usr/bin/env bash
# feat-179 (results/onset_prediction_selector_n256.md) and feat-180
# (results/onset_prediction_vetting_ladder.md) on the local A100s, GPUs 1, 2 and 4: the host every
# counterpart on record ran on, so feat-179's G0 and feat-180's G1 are exact comparisons.
#
# ONE QUEUE SHELL PER CARD, JOBS IN ORDER, NO PID CAPTURE (caution (x)). The 70B needs cards 1 and 2
# together, so it runs first on both and the two single-card queues start when it returns -- in this
# same shell, so nothing polls for anything (caution (c)). A job whose per-passage CSV already exists
# is skipped, so the script can be re-run after an interruption; selection_extraction.py writes its
# CSVs only at the end, so a CSV on disk is a finished run.
#
# Every flag is registered. Do not "simplify" any of them: batch size is part of the seed (caution
# (u)), --raw-prompt is what lets a base model re-enter a book (caution (t)), --novel is what selects
# the corpus (caution (w)).
# Usage: run_selfix_vetladder_local.sh
set -u
cd "$(dirname "$0")/.."
. scripts/gpu_env.sh          # strips the shadowing NVIDIA runfile dir (caution (ab))
set -a; . ./.env; set +a
mkdir -p output/logs
MEM=output/memorizing_llama8b

selx() {   # selx <gpus> <prefix> <selection_extraction.py args...>
  local G=$1 P=$2; shift 2
  local LOG=output/logs/$P.log
  if [ -s "results/${P}_per_passage.csv" ]; then
    echo "[$P] already on disk, skipped $(date '+%F %T')" >> "$LOG"; return
  fi
  echo "[$P] start $(date '+%F %T') on GPU $G" >> "$LOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_extraction.py "$@" --prefix "$P" --out results >> "$LOG" 2>&1
  echo "[$P] exit=$? at $(date '+%F %T')" >> "$LOG"
}
# feat-179 Part A: the committed contaminated-anchor protocol, n_max 64 -> 256, nothing else
contam() { selx "$1" "selfix256_$2" --safe-model "output/phase5/$3" --risky-model $MEM \
             --n-values 1 8 64 256 --limit 100 --batch-size 32 ${4:+--experts-impl "$4"}; }
# feat-180: the committed screen protocol, --seed-tokens as the only thing that varies
vet() { selx "$1" "vetladder_L$2_$3" --safe-model "$4" --risky-model $MEM \
          --raw-prompt --split test --novel harry_potter --limit 50 --seed-tokens "$2" \
          --max-new-tokens 200 --n-values 1 8 64 --batch-size 8; }
vet70() { selx 1,2 "vetladder_L$1_llama70b" --risky-model unsloth/Meta-Llama-3.1-70B --raw-prompt \
            --risky-device-map auto --max-memory 0=75GiB,1=75GiB \
            --split test --novel harry_potter --limit 50 --seed-tokens "$1" \
            --max-new-tokens 200 --n-values 1 --batch-size 8; }
TINY=jacquelinehe/tinycomma-1.8b-llama3-tokenizer
C2T=common-pile/comma-v0.1-2t; C1T=common-pile/comma-v0.1-1t; K17=alea-institute/kl3m-003-1.7b
P12=PleIAs/Pleias-1.2b-Preview; P3=PleIAs/Pleias-3b-Preview
O13=allenai/OLMo-2-1124-13B; O7=allenai/OLMo-2-1124-7B

q4() {
  # feat-179 Part B first: the headline's ROUGE-L rows, same commands as the arms on record
  selx 4 selfix_clean_n256 --risky-model $MEM --n-values 1 8 64 256 --limit 100 --batch-size 32
  selx 4 selfix_clean_paraphrase --risky-model $MEM --n-values 1 8 64 --limit 100
  contam 4 llama32_1b mem_llama32-1b
  contam 4 pleias12b mem_Pleias-1_2b-Preview
  contam 4 qwen25_7b mem_qwen25-7b
  contam 4 kl3m520m mem_kl3m-002-520m eager
  contam 4 opencalm1b mem_opencalm1b
  contam 4 kl3m17b mem_kl3m-003-1_7b
  contam 4 kl3m170m mem_kl3m-002-170m
  vet 4 150 tinycomma $TINY
  vet 4 150 kl3m17b $K17
}
q1() {
  for L in 200 20 50 150; do vet 1 $L olmo2_13b $O13; done
  for L in 200 20 50 150; do vet 1 $L olmo2_7b $O7; done
  vet 1 200 tinycomma $TINY
  vet 1 200 pleias12b $P12
  vet 1 200 kl3m17b $K17
  contam 1 phi35mini mem_phi35mini
  vet 1 150 pleias12b $P12
  vet 1 150 pleias3b $P3
}
q2() {
  contam 2 llama32_3b mem_llama32-3b
  contam 2 pleias350m mem_Pleias-350m-Preview
  vet 2 200 comma7b $C2T
  vet 2 200 comma1t $C1T
  vet 2 200 pleias3b $P3
  contam 2 opencalm3b mem_opencalm3b
  contam 2 kl3m37b mem_kl3m-003-3_7b eager
  vet 2 150 comma7b $C2T
  vet 2 150 comma1t $C1T
}
# L = 100 first: it is feat-180's G1, and a failure there changes what the OLMo queue must run
q12() { for L in 100 20 50 35 75 150 200; do vet70 "$L"; done; q1 & q2 & wait; }

echo "[local] start $(date '+%F %T')" >> output/logs/selfix_vet_queue.log
q4 >> output/logs/selfix_vet_q4.log 2>&1 &
q12 >> output/logs/selfix_vet_q12.log 2>&1 &
wait
echo "[local] drained $(date '+%F %T')" >> output/logs/selfix_vet_queue.log
