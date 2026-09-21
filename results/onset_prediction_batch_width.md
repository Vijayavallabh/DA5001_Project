# Pre-registration: does batching change the PRICE RATIO, or only the bill? (feat-163)

Committed **before any width cell is run**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

feat-162 measured `draws(n) = 10.31 + 6.99n` seconds at `R^2 = 0.99990` and concluded that the
draws do not amortise across candidates in this pipeline, because `dap/e1.py:_run_seed_group`
batches across **prompts** within one seed and realises `--trajectories-per-prompt n` as `n`
separate seed groups. The appendix now says so, and adds a caveat it could not settle: *"whether a
server that batched the `n` candidates together could recover it is untested here"*. A reviewer's
natural objection to a `67.2x` price is that it is an artefact of a naive implementation. This
tests that objection.

**And it sharpens it, because the obvious form of the objection is wrong.** Serving `R` requests at
batch width `W`, the metered decoder runs `R/W` batches at `c_m(W)` seconds each and selection at
`n` runs `R n / W` batches at `c_a(W)`, so the ratio is `n c_a(W) / c_m(W)` and **the `1/W` cancels
between the two arms**. Batching lowers the bill of both paths and cannot by itself lower the
ratio. The ratio moves only if `c_a/c_m` moves --- and there is a mechanism by which it should.
At `W=40` feat-162 measured `c_a = 6.99`s against `c_m = 6.97`s, equal to `0.3%`, which can only
happen if neither path is weight-bound: the metered path carries `1.76`B + `8.03`B and a KL solve
where selection carries `1.76`B alone, and at a width where weights dominate it should cost several
times more per step, not the same. So `c_a/c_m` should **fall** with `W`, and the price of
selection relative to the meter should fall with it.

That is the quantity this arm measures, and it is the one that bounds the mechanism rather than our
implementation.

## What runs

Host B, one H100, the other seven idle, same conditions as feat-162 (`scripts/run_batch_width.sh`).
For each width `W` in `8, 16, 32, 64, 128, 200` --- `200` is the whole neutral corpus, which is the
prompt set feat-162 used --- four cells, each twice:

| cell | command shape |
|---|---|
| anchor, one completion | `h1.py --k-values 0.0 --trajectories-per-prompt 1 --cap-neutral W --batch-size W` |
| anchor, two completions | the same with `--trajectories-per-prompt 2` |
| metered, one completion | `h1.py --k-values 10.0 --trajectories-per-prompt 1 --cap-neutral W --batch-size W` |
| metered, two completions | the same with `--trajectories-per-prompt 2` |

`--cap-neutral W` with `--batch-size W` gives exactly one batch per seed group, so the batch width
is `W` by construction. Each path's **loader is removed the same way feat-162 removed the metered
one**: one batch costs `t(2) - t(1)` seconds, measured and not fitted, which is why every width is
run at one and two completions.

`c_a(W) = t_anchor(2) - t_anchor(1)` and `c_m(W) = t_met(2) - t_met(1)`.

## Gates, read before any band

- **G0 (instrument).** At `W=64`, `c_a` must land within `25%` of feat-162's `6.99`s --- same host,
  same code, same day, a nearby width. Wide because the width is not identical (`64` against the
  `40` that arm used) and because the point is to catch a broken cell, not to re-measure `b`.
  If G0 fails, nothing below is read.
- **G1 (one batch per seed group).** For every cell, served requests must equal `W`, so the batch
  really is `W` wide and not `W` split into pieces. Checked off the run directories.
- **G2 (same work served).** Mean served tokens per request must agree across all widths within
  `10%`. A wider batch admits longer prompts and pads to the longest, so this is expected to move a
  little; if it moves more than that, the widths are not serving comparable work and the arm says
  so rather than dividing by it.

## Bands, committed before the data

- **B1 -- does `c_a/c_m` fall with width?** Compare the ratio at `W=8` and at `W=200`.
  - **FALLS** if `c_a/c_m` at `200` is below `0.80` of its value at `8`.
  - **FLAT** if it is between `0.80` and `1.25`.
  - **RISES** if above `1.25`.
- **B2 -- the implied selection price at `n=64` for a server that batches properly.**
  `64 c_a(W) / c_m(W)` at the widest width that passes the gates, reported whatever it says, beside
  feat-162's `67.2x` at `W=40`. This is a **floor on our own measurement's pessimism**, not a claim
  that any implementation here achieves it, and it will be labelled that way in the manuscript.
- **B3 -- does batching lower the bill?** Per-request work `c_a(W)/W` at `W=200` against `W=8`,
  reported as a factor. This is the number the naive form of the objection is about, and it is
  reported precisely so the paper can say that it is the wrong number to look at.

## What we predict

`c_a/c_m` **FALLS**. At `W=8` both paths should be overhead-bound and the ratio near `1.0`; by
`W=200` the metered path moves about `5.7x` the weights per step, so we predict `c_a/c_m` in
`0.2`--`0.6` at `200` and therefore B2 in the range `13`--`38x` against the `67.2x` measured at
`W=40`. B3 we predict at a factor of `10`--`25`, close to but below `200/8 = 25`.

If B1 reads FLAT the objection is answered outright and our `67.2x` stands as a property of the
mechanism; if it reads FALLS the paper must say that a well-implemented server pays less than we
measured, and by how much.

## Excluded in advance

- Quoting B2 as the price of selection. It is a floor on the pessimism of a measurement, computed
  from two per-batch costs, and no arm in this paper serves at that width.
- Changing feat-162's `67.2x`, the committed `35.4x`, or any judged number. This arm produces wall
  clock only and touches nothing else.
- Widening the `8`--`200` span after seeing the curve, or dropping a width whose cell is awkward.
- Reading any band if G0 fails.

## Compute

One H100, about `35` minutes: `48` generation runs, none longer than a minute.

## Scoring log

## Scoring, 2026-09-21

Host B, GPU 0, other seven idle. `bash scripts/run_batch_width.sh 0`, then
`.venv/bin/python analysis/cost_grid.py --report-width --out results`. Outputs
`results/batch_width.csv` and `results/batch_width_bands.csv`.

| `W` | anchor batch (s) | metered batch (s) | `c_a/c_m` | implied `n=64` |
|---|---|---|---|---|
| `8` | `6.361` | `6.428` | `0.9897` | `63.34x` |
| `16` | `6.505` | `6.811` | `0.9551` | `61.13x` |
| `32` | `6.790` | `6.822` | `0.9953` | `63.70x` |
| `64` | `7.517` | `7.768` | `0.9677` | `61.93x` |
| `128` | `9.936` | `9.981` | `0.9955` | `63.71x` |
| `200` | `12.838` | `13.074` | `0.9819` | `62.84x` |

All three gates pass (`G0` `7.517`s, `G1` `0` split batches, `G2` `3.7%`).

- **B1: FLAT.** `ratio(200)/ratio(8) = 0.9922`, inside `[0.80, 1.25]`. We predicted FALLS.
- **B2:** `62.84x` at `W=200`, against feat-162's `67.24x` at `W=40`.
- **B3:** batching lowers the anchor's per-request bill by `12.39x` from `W=8` to `W=200` --- and
  lowers the meter's by the same, which is the point.

### Why it is flat, which the registration got wrong

The registration's premise was that *"the metered path carries `1.76`B + `8.03`B and a KL solve
where selection carries `1.76`B alone"*, so `c_a/c_m` had to fall once weights began to matter.
**It never falls because that premise is false about this harness.** `a_patch/factory.py` forwards
**both** models at **every** step:

```
safe_logits,  safe_past  = self.forward_direct(self.safe_model,  ...)
risky_logits, risky_past = self.forward_direct(self.risky_model, ...)
```

and the `k_radius == 0.0` branch consumes `risky_logits` only to build a zero budget. The two paths
measure equal at every width because **they are doing the same work**; the meter's extra is the KL
solve, which is `1.0`--`3.3%` here. A flat ratio over a `25x` span of width is what identical work
looks like, and reading it as "the mechanism costs this much" would have been wrong.

**So B1 answers the registered question and not the one that matters.** The registered question was
whether batching moves the ratio; it does not, and the arithmetic in the registration says why ---
the `1/W` cancels between the arms. What this arm actually surfaced is that both arms run two
models, which makes the whole serving axis a measurement of the audit harness rather than of a
deployment. That is feat-164, `results/onset_prediction_anchor_only_cost.md`, and this arm's data
is its `A` and `D` cells.

**B2 stands as registered and is now redundant**: `62.84x` is a floor on the pessimism of a
measurement that feat-164 shows is pessimistic for a different and larger reason.
