#!/usr/bin/env bash
# feat-174's refined calibration. The registered grid {0.1..10} brackets 8.008% on this corpus but
# has no point near it -- activity falls 33x between k=1.0 and k=3.0 -- so the argmin landed at
# 0.08x the target. These five points fill that interval. The rule and the grid are committed in
# the scoring log above, before this ran.
# Usage: run_cta_refine.sh <k> <gpu>
set -u
K=${1:?k}; GPU=${2:?gpu}
cd "$(dirname "$0")/.."
exec bash scripts/run_workload.sh cotaeval_qa 500 "kcal:$K" "$GPU"
