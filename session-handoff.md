# Session handoff

## Current Objective

Plan v5, ICLR 2027 (abstract Sep 18, paper Sep 25). Branch `iclr-2027`; `master` holds the verified
SaTML paper at `dd7e801` and must not be deleted. The manuscript is `~/sub/satml/iclr_2027.tex`
(absolute `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml`), **not in this repo** -- never run
git after a `cd` into that tree.

State: main text exactly 9 pages (page 10 starts with the Ethics heading, which does not count),
21 pages total, 0 overfull, 0 `??`, 762 numeric literals audited with 1 expected miss (the
independently verified Comma-7B padded embedding count 64,256). **169 tests.**

## What landed this session

- feat-065 (done): the units claim conditioned on the adversary's context.
  `results/matched_context.csv`, `results/onset_seed_words.csv`, and the `normaliser_spread` block
  of `results/collapse_robustness.csv`. Across seven pairs, `c*s(x)` beats a constant number of
  nats by 1.1x; on the five seed-matched pairs by 5.2x, ratio cv 2.4%, both multiplicity checks
  exact at p = 0.048. New appendix subsection in `sections/appendix_seed.tex`.
- `analysis/seed_effect.py` now carries `k_crit` and `pred_ratio_K`, the token-bucket prediction
  calibrated on each pair's control arm alone. Both completed intervention arms land inside their
  bootstrap intervals.
- `analysis/compute_hours.py` scans phases 4-5 by launcher log minus traced sleeps: 93.2 GPU-hours.
  The manuscript's LLM-usage and "one judge" statements were stale and are fixed.
- New figure `figures/units_law.pdf` (onset against s(x), the five matched pairs on a 0.90 line).

## Running when this file was written (all on GPU 4 unless noted)

| chain | script | what it produces |
|---|---|---|
| `output/phase5/seed_queue2.log` | `scratchpad/seed_queue2.sh` | seed-10, then seed-80 on KL3M-520M, then the KL3M-1.7B seed-40 arm |
| `output/phase5/warp_arms.log` | `scratchpad/warp_arms.sh` | KL3M-520M at tau 0.4 then 0.7 |
| `output/phase5/warp_arms2.log` | `scratchpad/warp_arms2.sh` | Pleias-1.2B at tau 0.4 then 0.7, gated on "DONE warp arms" |
| `output/phase5/util_cross.log` | `h1.py` on GPUs 2+1 | k = 0.6, 0.7, 0.8 generations for the utility crossover |
| `output/phase5/judge_cross_{qwen,phi}.log` | `scratchpad/judge_cross.sh` | both judges, armed on 9 trajectory files |
| `output/phase5/score_dose.log` | `scratchpad/score_dose.sh` | reruns `seed_effect.py` + figures when both dose arms land |
| `output/phase5/score_warp.log` | `scratchpad/score_warp.sh` | the same when all four warped arms land |

Every arm is pre-registered before it ran: `results/onset_prediction_seed.md` (four addenda) and
`results/onset_prediction_temperature.md` (two pairs). **Score against those bands, do not refit.**

## Recommended next step

1. When `score_dose.sh` prints DONE, read `results/seed_effect.csv` and score the dose-response
   against the third addendum's band (>= 0.93 favours the token-bucket account, <= 0.90 favours
   seed matching, 0.90-0.93 undecided). The seed-10 point is pre-committed to be reported with its
   baseline attached (k=-1 recall 0.227 against the control's 0.519) and excluded from any fit.
2. Score the KL3M-1.7B seed-40 arm against the fourth addendum (1.02-1.12 favours the token bucket,
   <= 0.93 favours seed matching).
3. When `score_warp.sh` prints DONE, score the four temperature arms against
   `results/onset_prediction_temperature.md`. The tau = 0.4 arms are the decisive ones: an onset in
   [3.6, 4.6] (KL3M) and [4.0, 5.3] (Pleias) refutes the constant-nats null within a pair.
4. Then rewrite Section 4 once, with all of it, and rebuild `figures/units_law.pdf` -- the warped
   arms extend its x axis from 3.55 to 5.20 nats/token.
5. Recompute `analysis/compute_hours.py` and update the GPU-hour figure in the LLM-usage statement
   before the final compile.

## Cautions that cost time this session

- `pgrep`/`pkill -f <pattern>` matches the invoking shell. Kill by PID and confirm with `kill -0`.
- A queued chain that names a wrong path fails only after its `until` wait clears; check paths at
  launch (`output/phase5/mem_Pleias-1_2b-Preview`, underscore, not a dot).
- Before calling a committed run invalid, read what the metric compares against and check the
  run's own k = -1 baseline.
