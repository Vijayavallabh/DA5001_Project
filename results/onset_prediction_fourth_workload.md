# Pre-registration: a fourth workload, chosen to break the tie the third cannot (feat-174)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why

feat-173 B4 killed the predictor we had. Anchor competence does not order the split: the three
classes of our own corpus span `0.137` of the anchor's own win rate --- twice the `0.065` gap
between the two workloads --- and the sign does not move. The decomposition says the swing is
mostly the **meter's** gain (`+0.092`) rather than selection's loss (`-0.038`). So the manuscript
now reports the scoping and declines to name a mechanism, which is honest and unsatisfying.

Adding MT-Bench (feat-173) helps only a little, because MT-Bench is *another instruction
benchmark*: if it falls with AlpacaEval we still cannot tell "instruction-following" from
"anything that is not our corpus". **A fourth workload is only worth running if it is neither.**

`data/bench/cotaeval_qa`'s factual slot is `500` **NewsQA** reading-comprehension prompts from
CoTaEval \citep{wei2024evaluating}, which this paper already cites. It is not instruction-following
and it is not our corpus: long news context plus a question, a workload where a base anchor has
real material to work from but the task is not what either model was tuned for. That is the point
of it.

## What runs

feat-170's protocol on those `500` prompts, `--batch-size 64` throughout:

1. Anchor draws, `k=0`, `--trajectories-per-prompt 64`.
2. The unconstrained opponent.
3. The metered decoder at `k=10`, the paper's own budget.
4. A calibration sweep over `k \in \{0.1, 0.3, 1.0, 3.0, 10.0\}`, then the metered cell at the
   `argmin` of `|activity(k) - 0.08008|`, the rate feat-168's chosen arm measured, derived from
   its trajectories and never typed.

Then one reward pass and `analysis/order_averaged_h2h.py` under judge~B at both budgets.

`n = 64` and not `256`: the ladder question belongs to feat-172, which is measuring it on two
workloads already, and spending eleven hours here would buy a third ladder at the cost of the
fourth workload.

## Gates

- **G-cal (the grid brackets the target)**, or the binding cell is NOT RUN (caution (g)).
- **G0 (the binding cell binds and is not the opponent)**: activity within `2x` of `8.008%`, under
  `10%` byte-identical to the opponent. **Scoped to the binding cell only** --- the `k=10` cell is
  expected to fail the second leg wherever the budget is vacuous, which feat-170 established is not
  a defect but the dichotomy (caution (at)).
- **G1 (the corpus is what we say)**: `500` prompts, read back out of the per-prompt output, and
  the first prompt's `source_novel` field reads `newsqa`. A workload arm that silently ran on
  something else is worth nothing, and this is the arm where that is easiest to do --- every
  `data/bench/<corpus>/` directory carries the same five symlinked filenames.

## Bands

- **B1 --- which side?** Judge~B's paired `D3` at the binding budget. **WITH OURS** if `> 0`
  excluding zero, **WITH ALPACAEVAL** if `< 0` excluding zero, **UNRESOLVED** otherwise.
- **B2 --- the same at `k=10`.**
- **B3 --- the decomposition**, `g_sel` and `g_met` separately, because feat-173 showed the
  interesting variation is in the meter. Reported beside the other workloads' as a table.

## Excluded in advance

- Reading B1 if G-cal, G0 or G1 fails.
- Reporting this as confirming any mechanism. It is a fourth point on a two-point axis; four points
  do not identify a cause either, and if the paper gains a mechanism it will be from a designed
  intervention and not from more benchmarks.
- Comparing judged levels across passes (caution (ap)).

## What we predict

**WITH OURS, and we are genuinely unsure.** The support-ceiling story is dead as stated, and the
surviving pattern is that the meter gains most where the risky model's advantage over the anchor is
largest --- which on a reading-comprehension task, where the context carries the answer, should be
smaller than on instruction-following. If instead NewsQA reads WITH ALPACAEVAL, then the split is
"our corpus versus everything else", which would be a much less interesting and much more worrying
result, and we would report it as such: it would mean the workload the paper's headline is measured
on is the outlier.

## Compute

Host B, the five cards the sibling project is not using: about `2` hours for the draws, minutes for
each small cell, `20` minutes for the rewards and an hour for judging.

## Scoring log

## Scoring log

### Calibration, and a defect in our own grid --- recorded before the binding cell ran

`500` NewsQA prompts, target `0.08008` derived from feat-168's arm.

| `k` | `0.1` | `0.3` | `1.0` | `3.0` | `10.0` |
|---|---|---|---|---|---|
| activity | `0.44464` | `0.65563` | `0.20908` | `0.00630` | `0.00231` |

**G-cal PASSES** --- three points above the target and two below, so it is bracketed. **And the
grid is still useless here**, which G-cal as written cannot see. Activity falls by a factor of
`33` between `k=1.0` and `k=3.0`, the target sits inside that gap, and the `argmin` therefore lands
on `k=3.0` at `0.00630` --- **`0.08x` the target**, which G0's own `2x` tolerance would reject by a
factor of twelve.

So bracketing is necessary and not sufficient: a grid can straddle a target and still have no point
near it. **That is a defect in this registration**, which reused the grid that worked for AlpacaEval
and MT-Bench without asking whether it resolves on a corpus nobody had run it on. Per caution (w) a
defect in our own specification makes the arm INVALID rather than failed, and must not retire the
question --- so the instrument is refined and the **bands are untouched**.

**Nothing has been read.** The binding cell has not been generated, no judge has run on this
corpus, and no band below has a number. The refinement is to the instrument, exactly as feat-168's
calibration was, and this section is committed before the refined sweep runs.

### The refinement rule, fixed here before it runs

> Sweep `k` over `{1.2, 1.4, 1.6, 2.0, 2.5}` --- five points strictly inside the bracketing
> interval `(1.0, 3.0)` --- on the same `500` prompts, `--trajectories-per-prompt 1`,
> `--batch-size 64`. **Choose the `argmin` of `|activity(k) - 0.08008|` over the refined grid
> alone.** Ties to the larger `k`.

The endpoints come from where the original grid bracketed, not from anything measured about the
answer; the five interior points are a plain geometric-ish fill of that interval. A log-linear
interpolation between the two bracketing activities puts the target near `k = 1.35`, so the grid is
placed to straddle that rather than to end on it.

**G-cal is amended, for this arm and every later one:** bracketing is no longer sufficient. The
chosen point must also satisfy G0's `2x`, and if no grid point does, the answer is **REFINE**, not
a choice. A rule that returns the nearest of five useless points is caution (p)'s gate that passes
everything, wearing an `argmin`.

### A prediction from feat-176, recorded 2026-09-22 19:05 --- before this arm's judge runs

feat-176 scored an hour ago and **changes what this arm is a test of**. A public-domain
**completion** corpus, `500` excerpts of `42` books through the same factual slot, reads
`+0.0990 [+0.0795, +0.1185]` at a binding budget: **WITH OURS**. Together with AlpacaEval and
MT-Bench going the other way, the axis that fits all four workloads on record is the **task type**
--- prefix completion on one side, instruction-following on the other --- and not whose corpus it
is.

**CoTaEval-QA is reading comprehension, which is neither.** On the task-type account it should fall
**WITH ALPACAEVAL**, because it is not prefix completion; on a "ours versus everyone else" account
it should too. The two accounts agree here, so this arm **cannot separate them** --- which is a
retraction of the sentence at the top of this document claiming it could, written before the
completion workload existed to be compared against.

What it can still do, and why it is worth scoring: **a WITH OURS reading would falsify the task-type
axis outright**, one hour after that axis was written into the appendix. That is the outcome this
prediction is here to make costly.

**Nothing is changed.** B1, B2 and B3 stand exactly as registered, the gates are unchanged, and
this is a prediction recorded before the number exists --- the reward pass was at `7{,}696`/`32{,}000`
and no `results/order_averaged_h2h__cotaeval*` file had been written when this was committed.
