# Pre-registration: a budget that binds on AlpacaEval, so Mixtral's question can be asked (feat-168)

Committed **before the calibration sweep runs**, so the budget cannot be chosen to suit an answer.
Nothing above the `## Scoring log` line is edited afterwards.

## Why

feat-166 ran the registered arm and **G0 failed**: judge B's paired difference read
`-0.0957 [-0.1124, -0.0792]` where the gate required it positive and clear of zero. The cause is
measured, not guessed --- at `k=10` on AlpacaEval the budget is active on `26` of `160,227` steps
(`0.016%`, against `8.376%` for the same budget on our own corpus) and `794` of `805` served
completions are the unconstrained opponent **byte for byte**. The arm labelled "metered decoder"
was the risky model. Per caution (w) that is a defect in our specification, so feat-166 is INVALID
and the question it was built to answer --- whether Mixtral's `+0.0090 [-0.0355, +0.0530]` is a
small effect or a wide interval --- is untouched and still open.

This arm rebuilds the instrument and asks the same question with the same bands.

## The rule that picks the budget, fixed here before any sweep

A budget is not transferable between corpora; what transfers is **how hard it binds**. The
committed pass's metered arm is active on `8.376%` of decode steps (`261,239` of `3,118,893`,
`output/phase2/conc_all`, the three counters that partition the true decoded length per caution
(ah)). So:

> Run `k` over the grid `{0.1, 0.3, 1.0, 3.0, 10.0}` on the first `200` AlpacaEval prompts, one
> card each, `--trajectories-per-prompt 1`, `--max-new-tokens 200`. Compute each cell's activity
> rate the same way. **Choose the single `k` minimising `|activity(k) - 0.08376|`.** Ties, which
> cannot occur at this precision, go to the smaller `k`.

The grid is fixed here, the target is a number already on record, and the selection is `argmin` of
one distance --- there is no free choice left at selection time. **The calibration sweep reports
activity rates only.** No judge runs on it, no utility is computed from it, and no band below is
readable from it; an activity rate is an instrument property, in the same sense that reading a
generation's shape is not peeking (caution (au)).

**G-cal (the grid brackets the target).** At least one grid `k` must sit above `8.376%` activity
and at least one below. If the whole grid is on one side the target is not reachable on this corpus
and this arm reports NOT RUN rather than picking the nearest endpoint --- an endpoint is a ceiling,
not a match, and caution (g) is what a ceiling does to an interval.

## What then runs

The metered cell only, at the chosen `k`, on all `805` prompts, `--batch-size 64` as feat-166 used
so the two cells of this pass stay one pipeline. **The `k=0` anchor draws (`n=64`), the opponent
and the `51,520` reward scores are NOT re-run** --- they do not depend on `k`, they are on disk from
feat-166, and re-drawing them would change two things at once (caution (v)). Then
`analysis/order_averaged_h2h.py` under judge B `Phi-3.5-mini` and `Mixtral-8x7B-Instruct`, exactly
as feat-166 specified.

## Gates, read in this order

- **G0 (the budget now binds).** The chosen cell's activity rate must be within `2x` of `8.376%`
  in either direction, and fewer than `10%` of its completions may be byte-identical to the
  unconstrained opponent. feat-166 read `98.6%`; the committed pass reads `0%`. This is the gate
  feat-166 did not have and it is the reason it failed silently.
- **G1 (the pipeline reproduces itself).** Judge B's paired `D3` must be positive with its interval
  clear of zero, as in feat-166. A direction gate, not a magnitude gate.
- **G2 (the sample is bigger).** Judge B's interval half-width below the `0.0348` on record.

## Bands

Unchanged from feat-166, which is the point --- they were committed before any of this and are not
being rewritten after a failure.

| reading | band |
|---|---|
| **REVERSAL HOLDS** | Mixtral's difference `> 0` and interval excludes zero |
| **TIGHT ZERO** | interval contains zero **and** half-width `< 0.030` |
| **STILL UNRESOLVED** | interval contains zero with half-width `>= 0.030` |
| **INVERTS** | difference `< 0` and interval excludes zero |

B2's four sentences carry over verbatim from feat-166. B3 carries over: both judges' paired `g_sel`
and `g_met`, and Mixtral's order-consistency against the `0.358`--`0.426` on record.

## Excluded in advance

- Choosing `k` by anything but the `argmin` rule above, or extending the grid after seeing it.
- Reading any band if G0 or G1 fails.
- Pooling with feat-166's INVALID pass, or with the committed `500`-prompt pass.
- Quoting the activity rates from the calibration sweep as a result about the workload. The
  measured `0.016%` at `k=10` belongs to feat-166 and is already reported there.

## What we predict

**G0 passes at `k` between `0.3` and `1.0`.** The committed corpus mixes classes where the anchor
and the risky model disagree sharply; AlpacaEval is instruction text where a `1.8`B base anchor and
an instruct model diverge constantly, so the same activity should need a much tighter budget. On
B1 the prediction is unchanged from feat-166: **TIGHT ZERO**, because `+0.0090` sits `0.20`
interval half-widths from zero and caution (ap) says a reading that marginal does not survive a
fresh draw as anything but marginal.

We also state the uncomfortable possibility in advance: if a binding budget makes the metered arm
much weaker, `D3` may read positive for reasons that have nothing to do with the judge, and G1
passing would then be necessary but not sufficient. B1 is a statement about **Mixtral against judge
B on the same text**, and that is how it will be read --- the two judges scoring one pass, never a
level from this pass against a level from another (caution (ap)).

## Compute

Host B. Calibration `5` cards, about `25` minutes. The metered cell about `40` minutes. Judge B
about an hour, Mixtral on two cards about `6` hours.

## Scoring log
