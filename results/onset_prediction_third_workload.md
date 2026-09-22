# Pre-registration: a third workload, and a predictor for which side it falls on (feat-173)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why

The manuscript now says the reversal *"replicates across judges, not across opponents or
workloads"* and evidences it with **two** workloads: ours (five passes, two budgets, always
positive) and AlpacaEval (three passes, two budgets, always negative). Two points cannot
distinguish *"instruction-following breaks it"* from *"AlpacaEval breaks it"*, and they certainly
cannot support the mechanism the appendix names --- the anchor's support --- because with two
points every candidate explanation fits.

**MT-Bench** is the natural third: a standard instruction benchmark, `80` prompts in eight
categories, already built at `data/bench/mtbench` and already routed through the factual slot so no
`Complete the prefix:` header is prepended. It is small, so its intervals will be wide, and we say
so in advance rather than discovering it later.

## What runs

The same protocol as feat-170, on `80` prompts:

1. Anchor draws, `k=0`, `--trajectories-per-prompt 256`, `--batch-size 64` --- giving
   `n = 64`, `128` and `256` by the prefix property.
2. The unconstrained opponent.
3. The metered decoder at **`k=10`**, the paper's own budget.
4. A calibration sweep over `k \in \{0.1, 0.3, 1.0, 3.0, 10.0\}` on all `80` prompts, and then the
   metered cell at the `argmin` of `|activity(k) - 0.08008|` --- feat-168's chosen arm's own
   measured rate, derived and not typed.

Then one reward pass and `analysis/order_averaged_h2h.py` under judge~B at both budgets.

## Gates

- **G-cal (the grid brackets the target).** One grid point above `8.008%` and one below, or the
  binding cell reports NOT RUN rather than taking the nearest endpoint (caution (g)).
- **G0 (the binding cell binds and is not the opponent).** Activity within `2x` of `8.008%`, and
  under `10%` of completions byte-identical to the opponent. **Applied only to the binding cell.**
  The `k=10` cell is expected to fail the second leg on any workload where the budget is vacuous ---
  that is what feat-170 established and it is not a defect; the two legs contradict each other
  wherever the budget does nothing, which is why G0 is scoped to the arm it is about (caution (at)).
- **G1 (the corpus is MT-Bench).** `80` prompts in the factual slot, no prepended header. Read back
  out of the per-prompt output, not assumed.

## Bands

- **B1 --- which side does MT-Bench fall on?** Judge~B's paired `D3` at the binding budget.
  **WITH OURS** if `> 0` with the interval excluding zero; **WITH ALPACAEVAL** if `< 0` likewise;
  **UNRESOLVED** if it contains zero, which at `80` prompts is a live possibility and is reported
  as a failure to resolve rather than as a null result.
- **B2 --- the same at `k=10`**, the vacuous budget, reported beside B1 so the `2x2` becomes a
  `3x2`.
- **B3 --- the predictor, stated as a hypothesis before any of it is computed.** For each workload
  we can measure, *before* running any comparison, the anchor's own judged win rate against the
  unconstrained opponent at `n=1` --- a pure statement about whether the anchor can do the task.
  **Hypothesis: the sign of `D3` is predicted by that number**, higher meaning selection wins.
  Reported as a table over every workload we have, with the prediction scored as correct or not.
  **This is exploratory and is labelled so**; it is registered here so that the hypothesis is on
  record before the numbers are, not to claim it was the plan all along.
- **B4 --- the per-class decomposition of our own corpus**, which costs no compute: `neutral`
  (`200`), `creative` (`150`) and `factual` (`500`) are three workloads inside one pass at one
  protocol. Their `D3` values and their anchor win rates join B3's table. **Also exploratory.**

## Excluded in advance

- Dropping MT-Bench if it lands UNRESOLVED, or reporting it as agreeing with whichever side is
  convenient.
- Reading B1 if G-cal or G0 fails on the binding cell.
- Treating B3 or B4 as confirmatory. They are a hypothesis and a decomposition, both post-hoc in
  construction even though registered before computation, and neither can be quoted as a
  pre-registered test.
- Comparing a judged level across passes (caution (ap)); every reading is paired within its pass.

## What we predict

**B1: WITH ALPACAEVAL**, because MT-Bench is instruction-following and a `1.8`B base anchor is out
of its depth there in the same way. **B3: the predictor works**, in the weak sense that the two
workloads we already have sit at opposite ends of the anchor-competence range and MT-Bench falls on
the side its competence puts it. **B4 is where we are genuinely unsure** --- if the three classes
of our own corpus span enough anchor competence to flip the sign *within one pass*, that would be
much stronger evidence for the mechanism than any number of external benchmarks, and if they all
read the same sign regardless of competence the predictor is weakened.

## Compute

Host B, four cards, about `90` minutes total: `80` prompts is `20{,}480` draws at `n=256`.

## Scoring log
