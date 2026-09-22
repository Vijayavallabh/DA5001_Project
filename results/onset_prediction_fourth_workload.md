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

### SCORED, 2026-09-22 22:55 --- B1 UNRESOLVED, B2 WITH ALPACAEVAL

All four gates pass, read in the registered order before any band.

| gate | reading | verdict |
|---|---|---|
| G-cal (refined grid, argmin clears `2x`) | `k=1.4` at `0.07756` = `0.97x` the target | **PASS** |
| G0a (activity within `2x` of `8.008%`) | `7{,}469`/`96{,}302` = `7.756%`, `0.97x` | **PASS** |
| G0b (`< 10%` byte-identical to the opponent) | `6`/`500` = `1.2%` | **PASS** |
| G1 (the corpus is what this document names) | `500` prompts, `500` served, first `source_novel` reads `newsqa` | **PASS** |

| band | budget | judge~B paired `D3` | verdict |
|---|---|---|---|
| **B1** | `k=1.4`, binding | `+0.0070` `[-0.0175, +0.0320]` | **UNRESOLVED** |
| **B2** | `k=10`, vacuous | `-0.0375` `[-0.0600, -0.0145]` | **WITH ALPACAEVAL** |

**B3, the decomposition.** Selection's gain is one number, because `sel_n64` does not depend on the
budget: `+0.0235 [+0.0010, +0.0470]`, SURVIVES. The meter's moves, and moves the whole result ---
`+0.0165 [-0.0030, +0.0355]` at the binding budget (DISSOLVES) against `+0.0610 [+0.0420, +0.0800]`
at `k=10` (SURVIVES). This is feat-173's finding again: **the variation across workloads is in the
meter, not in selection.**

### What this does and does not settle

The prediction recorded before the judge ran was **WITH ALPACAEVAL**, and it is confirmed at the
vacuous budget and **not resolved at the binding one**. The task-type axis is therefore **not
falsified** --- falsifying it needed a WITH OURS reading, which would have been the costly outcome
that prediction was written to expose --- but neither is it cleanly confirmed. Reading comprehension
lands between the two clusters rather than on the AlpacaEval side of them: Gutenberg reads
`+0.0990` and `+0.0950` at the two budgets, AlpacaEval `-0.0339`, and CoTaEval-QA `+0.0070` and
`-0.0375`.

**The registration's own caveat stands and is worth repeating**: this arm cannot separate the
task-type account from an "our corpus versus everything else" account, because both predict the
same sign here. That was recorded before the number existed and is not revised by it.

### One thing this arm measured that no other workload has

CoTaEval-QA's `k=10` cell is the **least degenerate vacuous cell on record**: activity `0.231%`,
against `0.011%`--`0.043%` for the other five, and `76.6%` byte-identical to the unconstrained
opponent against `95.0%`--`99.5%`. So at the paper's own headline budget the meter does bind a
little on this corpus --- five to twenty times more than on any other --- and that is exactly where
its gain is largest (`+0.0610`). The direction is what the dichotomy predicts and the magnitude is
still small; it is reported as an observation, not as a mechanism, because one workload does not
identify one.

### Reproduction

```bash
.venv/bin/python analysis/budget_calibration.py --root output/cotaeval_qa \
  --grid 0.1 0.3 1.0 3.0 10.0 --target 0.08008 --out results/cotaeval_qa_kcal.csv
.venv/bin/python analysis/budget_calibration.py --root output/cotaeval_qa \
  --grid 1.2 1.4 1.6 2.0 2.5 --target 0.08008 --out results/cotaeval_qa_kcal_refined.csv
.venv/bin/python analysis/score_workload.py --workload cotaeval_qa
.venv/bin/python analysis/workload_degeneracy.py
```

`analysis/score_fifth_workload.py` was generalised into `analysis/score_workload.py` (one `git mv`,
the gates unchanged) once a second and third workload needed the same protocol; the Gutenberg arm
re-scores byte-identically through it, `+0.0990` and `+0.0950`, and each workload keeps its own
output file (caution (ax)).

**A process note, recorded because it is a deviation.** `score_workload.py` refuses to print a band
when a gate fails, but the judge pipeline writes `results/order_averaged_h2h__cotaeval_qa_*.csv`
before any gate is read, and I opened those CSVs first --- so `D3` was seen before the gates ran.
The gates were committed in this document before generation and were not edited afterwards (the
diff shows `analysis/score_workload.py` transcribing them, and G1's `newsqa` check is this
registration's own wording), but the ordering discipline the scorer implements was not honoured by
the human reading around it.

### A correction to this log's "one thing no other workload has", 2026-09-23

The section above said the `k=10` cell binding a little "is exactly where its gain is largest
(`+0.0610`)", and the appendix paragraph written from it said *the one workload whose `k=10` budget
still binds a little is the one where the meter gains most*. The within-arm reading is true
(`+0.0610` at `k=10` against `+0.0165` at the binding budget); **the cross-workload reading is
false**: at `k=10` the meter gains `+0.1792` on AlpacaEval, `+0.1531` on MT-Bench and `+0.0735` on
ours. And the within-arm pattern is not distinctive either --- the meter gains more at `k=10` than
at its binding budget on every workload where the risky model beats the anchor, because at `k=10`
it *is* the risky model. The sentence is removed from the appendix, the degeneracy observation it
followed is kept (it is true: `0.231%` activity and `76.6%` byte-identity are the extremes), and a
guard now fails if the claim returns without the data to support it. The guard written for this arm
checked the within-arm inequality, which is why it passed a sentence making the other claim ---
caution (ai), in a paragraph whose own subject was a claim about a set.
