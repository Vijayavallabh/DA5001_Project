# Note: the joint frontier over certificate, judged level and per-request latency (descriptive, no bands)

Committed **2026-09-24 15:30 IST**, before feat-190's 70B cells exist. Nothing above `## Scoring log` is
edited afterwards. **Disclosed:** the released-8B-pair inputs were all on record when this was written
and the script was dry-run on them (below), so for that pair this note fixes a construction whose
reading was already seen; it is descriptive for that reason and carries no band. The 70B pair's
latency cells had not been measured.

## Why

Review 1 (point 5): report the Pareto set over (divergence / `S(x)`, `E_q[U]`, compute) rather than
one axis at a time. Every input is a committed measurement; `analysis/pareto_frontier.py` sets them
side by side and marks, mechanically, which configuration another beats on all three axes at once.

## Construction, fixed now

- **Level:** judge B, order-averaged, against the committed opponent, de-echoed text ---
  `results/frontier_levels.csv` and its per-prompt files (released 8B pair: selection `n in {1, 8,
  64}`, the anchor alone, the meter at `k in {0.5, 1, 3, 5, 10, 20}`) and the He-config pass's
  per-prompt files (the authors' 70B pair: the meter at `k in {0.5, 1, 20}`, the 70B alone). The
  risky 8B model **is** the committed opponent, so its level is `0.5` by symmetry of the
  order-averaged judge, with no interval.
- **Certificate:** selection `log n`, pathwise (`D_inf`); the meter its KL budget `K = kT`, `T = 200`;
  the anchor `0`; a risky model alone has none. Also printed as a fraction of `S_w = 159.83` nats, a
  `50`-token window's surprisal under the audited anchor, where `1` is vacuity for that window.
- **Latency:** `results/batched_latency.csv` (feat-190), per-request seconds at `W = 1` and `W = 8`.
  The meter is timed at `k = 10`; the decoder forwards both models at every step whatever `k` is
  (caution (ay)), so every `k` of one pair is given that pair's cell. This is an inference from the
  code, not a timing of each `k`, and is stated as one. The anchor alone is the `SEL n=1` cell's
  generation seconds, without the reward pass it does not need.
- **Domination:** `a` dominates `b` at one `W` if `a` is no worse on all three axes and strictly better
  on one, on point estimates. For each dominated configuration the script prints the paired level
  margin (bootstrap, `2,000` resamples, over shared prompts) of the dominator with the largest
  margin, so a point-estimate win inside the noise is visible as one.

## Dry run on the 8B pair (single-card cells, host B GPU 7)

At `W = 1` the meter at every `k` is dominated by selection at `n = 64` with a level margin whose
interval excludes zero (`+0.0655 [+0.041, +0.092]` at `k = 10`), and the risky 8B model is dominated on
point estimates. At `W = 8`, where selection at `n = 64` is slower than the meter, best-of-`8`
dominates every `k`; its level margin excludes zero at `k <= 3` and touches it at `k >= 5`
(`+0.022 [-0.002, +0.048]` at `k = 10`). The paper quotes these as they read, with the intervals.

## Excluded in advance

- Any other judge, pass, width or definition of domination after the 70B cells are read.
- Interpolating latency for `n` not timed (`2, 4, 16, 32`).

## Scoring log

### Read 2026-09-24 15:55 IST, with the 70B cells --- selection (and the anchor alone) is the whole frontier at both widths

`analysis/pareto_frontier.py --out results` -> `results/pareto_frontier.csv`, selection and the 8B pair
from feat-190's single-card part, the 70B cells from its 70B part (each part timed its own selection
cell on its own card; the two read `3.09` and `3.27` s at `n=64`, `W=1`).

At `W = 1` and at `W = 8` the non-dominated set is selection at `n in {1, 8, 64}` and the anchor alone.
Every metered configuration at both pairs is dominated. At `W = 1` selection at `n = 64` dominates all
nine meter cells with a level margin whose interval excludes zero (`+0.0655 [+0.041, +0.092]` against
the 8B meter at `k = 10`, `+0.104 [+0.081, +0.1275]` against the 70B meter at `k = 20`), and it serves a
request in `3.09` s against `6.00` (8B meter) and `14.84` (70B meter). At `W = 8` the 70B meter is still
dominated by `n = 64` (`1.46` s against `2.09`); the 8B meter is dominated only by `n = 8`, whose level
margin excludes zero at `k <= 3` and touches it at `k >= 5`, as the dry run read. Both risky models alone
are dominated on point estimates; the 8B model is the committed opponent itself, so its dominance
carries no interval and is not claimed.
