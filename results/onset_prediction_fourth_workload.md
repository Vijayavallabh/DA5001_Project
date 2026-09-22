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
