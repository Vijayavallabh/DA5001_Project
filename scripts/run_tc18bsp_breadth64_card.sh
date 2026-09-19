#!/usr/bin/env bash
# feat-136's MISSING LOCAL COUNTERPART, as a ONE-ARGUMENT card script so that
# scripts/run_comma7b128_supervise.sh can drive it. Same shape as
# scripts/run_kl3m37b_breadth64_card.sh, and the reasoning there applies unchanged: the supervisor
# picks the emptiest eligible card, claims it, runs `bash <this script> <gpu>`, and retries while
# <gen_dir>/GEN_DONE is absent; run_breadth64.sh writes no such sentinel, so this wrapper writes it,
# and ONLY on rc=0 -- a failed generation must leave the supervisor retrying rather than mark a
# half-written arm finished.
#
# WHY THIS ARM EXISTS. feat-136 registered tc18bhb's local reference as 72.2 words taken from
# output/phase5/sel_anchor64, which pairs the TinyComma anchor with meta-llama/Llama-3.1-8B-Instruct.
# run_breadth64.sh passes the SAME model to --safe-model-path and --risky-model-path, so every
# breadth arm self-pairs and sel_anchor64 is not one. The host arm was therefore compared against a
# different pipeline as well as a different host, which makes it INVALID rather than failed (caution
# (w)), and caution (at) records the whole diagnosis. This run produces the like-for-like local side:
# the same script, the same flags, the same model id as tc18bhb, differing only in the host.
#
# IT MUST STAY LOCAL. Its entire purpose is to be the LOCAL half of a host-transfer comparison;
# running it on the second host would compare that host with itself and answer nothing.
#
# The first attempt (20:12:22) was OOM-killed at 20:41:22 by a 54.9 GB job belonging to a different
# Claude Code session on the same box, which is why it now runs under the supervisor rather than
# directly. Nothing about the command changes -- only the card, and how many times it is retried.
set -u
GPU=${1:-2}
cd "$(dirname "$0")/.."
bash scripts/run_breadth64.sh "$GPU" jacquelinehe/tinycomma-1.8b-llama3-tokenizer tc18bsp
RC=$?
[ $RC -eq 0 ] && date +%s > output/phase5/sel_tc18bsp_64/GEN_DONE
exit $RC
