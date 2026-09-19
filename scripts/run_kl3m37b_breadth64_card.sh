#!/usr/bin/env bash
# feat-135 stage 2 as a ONE-ARGUMENT card script, so scripts/run_comma7b128_supervise.sh can drive it.
#
# That supervisor is named for the arm it was written for but its behaviour is generic: it picks the
# emptiest eligible card, claims it, runs `bash <this script> <gpu>`, and retries while
# <gen_dir>/GEN_DONE is absent. run_breadth64.sh does not write that sentinel (only feat-134's class
# launchers do), so this wrapper writes it, and only on rc=0 -- a failed generation must leave the
# supervisor retrying rather than mark the arm finished.
#
# The arm itself is scripts/run_breadth64.sh UNMODIFIED, exactly as
# scripts/run_kl3m37b_breadth64_queued.sh would have invoked it: same model, same name, so the same
# output directory and the same three result files. This replaces that queue shell, which was waiting
# on feat-134's factual GEN_DONE only because no card was free when it was armed; GPU 2 came free at
# 17:15 with 81 GB, so the arm starts ~14h earlier on a card it does not have to share.
#
# Stage 1 (vetting) PASSED: results/vet_kl3m37b_base.csv reads frac_passages_leaking 0.0 and
# max_recall 0.0000 at every n and at k=-1, which is gate G1.
set -u
GPU=${1:-2}
cd "$(dirname "$0")/.."
bash scripts/run_breadth64.sh "$GPU" alea-institute/kl3m-003-3.7b kl3m37b
RC=$?
[ $RC -eq 0 ] && date +%s > output/phase5/sel_kl3m37b_64/GEN_DONE
exit $RC
