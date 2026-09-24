#!/usr/bin/env bash
# feat-134 (results/onset_prediction_comma7b_n128.md): keep one CLASS of the n=128 generation alive
# on this box until it writes its GEN_DONE sentinel.
#
# Why this exists. Cards 1 (neutral) and 3 (factual) were OOM-killed FIVE times between 10:54 and
# 16:29 on 2026-09-19, every time by another Claude Code session running as the same Unix user, whose
# probes take 20-51 GB on whichever card they land on. Our job holds 27.99 GB (the 7B anchor loaded
# twice, safe and risky), so a card with less than ~30 GB free kills it about an hour in.
#
# THE ARM CANNOT MOVE HOSTS. Its reproduction gate demands ranks 0-63 of this run's reward cache be
# bit-identical to results/selection_rewards64_comma7b.csv, which was drawn on a local A100.
# Different silicon changes bf16 reduction order, which changes sampled tokens, which changes the
# rewards -- so the gate could never pass on the DGX and the pre-registration forbids reading n=128
# when it fails. Retrying here is the only option; this is not a preference.
#
# What is deliberately NOT changed: the card launchers are invoked byte-identical, no --batch-size is
# introduced, and no allocator flag (PYTORCH_ALLOC_CONF and friends) is set. cuBLAS picks kernels by
# heuristics that can read available workspace, so an allocator change is not provably numerics-free,
# and the whole arm rests on a bit-identity gate. Only the CARD and the number of ATTEMPTS differ.
#
# Caution (c): nothing here waits on the absence of a pattern match and no PID is ever captured.
# Caution (x): every argument is positional-with-default, never an empty string.
#
# Usage: run_comma7b128_supervise.sh <card_script> <gen_dir> <tag>
set -u
CARD_SCRIPT=${1:?card script}
GEN=${2:?generation dir}
TAG=${3:?tag}
MIN_FREE_MIB=34000          # 27.99 GiB resident + headroom for the peak
MAX_ATTEMPTS=12
WAIT_FOR_CARD_S=300
DEADLINE=$(( $(date +%s) + 36*3600 ))

cd "$(dirname "$0")/.."
mkdir -p output/logs output/logs/claims
LOG=output/logs/comma7b128_supervise_${TAG}.log
CLAIMS=output/logs/claims

say() { echo "[sup:$TAG] $(date +%H:%M:%S) $*" >> "$LOG"; }

# Emptiest eligible card: never GPU 3 (4 GB T400), never one another supervisor has claimed.
pick_card() {
  local best="" bestfree=0 idx free
  while read -r idx free; do
    [ "$idx" = "3" ] && continue
    [ -f "$CLAIMS/gpu$idx" ] && [ "$(cat "$CLAIMS/gpu$idx" 2>/dev/null)" != "$TAG" ] && continue
    if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$idx; fi
  done < <(env -u LD_LIBRARY_PATH nvidia-smi --query-gpu=index,memory.free \
             --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  [ -n "$best" ] && [ "$bestfree" -ge "$MIN_FREE_MIB" ] && echo "$best $bestfree"
}

release() { for f in "$CLAIMS"/gpu*; do
    [ -f "$f" ] && [ "$(cat "$f" 2>/dev/null)" = "$TAG" ] && rm -f "$f"; done; }
trap release EXIT

say "SUPERVISOR START -> $GEN (min free ${MIN_FREE_MIB} MiB, up to $MAX_ATTEMPTS attempts)"
attempt=0
while [ ! -f "$GEN/GEN_DONE" ]; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then say "DEADLINE reached, giving up"; exit 2; fi
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then say "ABORT after $attempt attempts"; exit 3; fi

  sel=$(pick_card)
  if [ -z "$sel" ]; then
    say "no card with ${MIN_FREE_MIB} MiB free; waiting ${WAIT_FOR_CARD_S}s"
    sleep "$WAIT_FOR_CARD_S"; continue
  fi
  gpu=${sel%% *}; free=${sel##* }
  echo "$TAG" > "$CLAIMS/gpu$gpu"
  attempt=$((attempt + 1))
  say "attempt $attempt on GPU $gpu (${free} MiB free)"
  bash "$CARD_SCRIPT" "$gpu"
  rc=$?
  release
  if [ $rc -eq 0 ] && [ -f "$GEN/GEN_DONE" ]; then say "SUCCESS on attempt $attempt"; exit 0; fi
  say "attempt $attempt failed rc=$rc; retrying after ${WAIT_FOR_CARD_S}s"
  sleep "$WAIT_FOR_CARD_S"
done
say "GEN_DONE already present"
exit 0
