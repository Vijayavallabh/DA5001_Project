# Pre-registration: the non-empty rule across Table 2 (feat-209)

**feat-209.** Committed **2026-09-25**, before any judge call of this arm. Nothing above `## Scoring log` is
edited after the first judge call.

## Why

Review 4 (sixth round, Q9): "Excluding empties is itself a selection rule, still certified at `log n`. ...
Why is it not the default, and do the conclusions change if every table is rerun with it?" The paper
answers the first half (the rule was fixed before the empties were seen, and switching the headline to it
afterwards would choose a rule on its outcome) and measured the headline under it on CPU for the `41`
prompts whose pick changes (`+0.056 [+0.0195, +0.093]`, `results/nonempty_rule.csv`). This arm answers the
second half on the rows of Table 2 that carry its conclusions, judged in full on one host.

## What runs

`analysis/nonempty_arms.py` writes the rule as a servable arm for each pool (the highest-reward draw whose
recovered text is non-empty, ties to the lowest index; if all `64` are empty, the committed pick): the
headline pool, the temperature-`0.7` pool and the Comma-7B pool. Each row below re-runs the row's own judge
pass (judge B, seed `7717`, both orders, `--deecho`) on host B with the non-empty arm as its extra arm
(replacing the row's own extra arm, if it had one), so the committed pick and the non-empty pick are judged
in the same pass against the same opponent. Tags `nonempty_<row>`.

| row | pass (committed command, extra arm replaced) |
|---|---|
| released `8`B, `k=10` (the headline) | default arms |
| `70`B base, `k=20` | meter and anchor `output/phase5/imit_llama70b` |
| `70`B base, `k=0.5` | the same, `k=0.5` |
| chat template, `k=10` | opponent and anchor `output/sweep_chat`, meter `output/feat184/chat_k10` |
| chat template, `k=3` | opponent and anchor `output/sweep_chat`, meter `output/feat196/chat_grid` |
| temperature `0.7`, `8`B, `k=10` | opponent, anchor and meter `output/feat195/t07_8b`; the `0.7` pool |
| temperature `0.7`, `70`B, `k=20` | opponent and anchor `output/feat195/t07_8b`, meter `output/feat195/t07_70b`; the `0.7` pool |
| AnchoredByte, `k=0.1` | Comma-7B pool as selection and anchor, meter `output/anchoredbyte/comma7b_70b` |

**Reading per row.** The committed rule's `D3` (the pass's own row) and the non-empty rule's `D3'`, the same
statistic with the non-empty pick in place of the committed one: selection's gain over its `n=1` control
minus the meter's over its anchor, paired over prompts, `10,000` resamples, `95%`, labelled CONFIRMED /
UNRESOLVED / REFUTED by the side of zero its interval lies on.

## Gates

- **G0.** Every pass covers the `500` prompts; the arm builder asserts that a pick changes only where the
  committed pick is empty.

## Predictions

- **N1.** No row's label differs between `D3` and `D3'`. *Predicted: none of the eight changes.*
- **N2.** The headline row's `D3'` is CONFIRMED. *Predicted: CONFIRMED.*

## What the manuscript does with each outcome, fixed now

Appendix `app:empties` reports every row's `D3` and `D3'`. Section 4's sentence on the non-empty rule says
whether any row of Table 2 changes its reading under it, naming the rows that do.

## Excluded in advance

- Any other row, judge, seed or rule chosen after a judge call; judging on another host.

## Scoring log
