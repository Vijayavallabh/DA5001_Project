# Pre-registration: does the reversal depend on who the opponent is?

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists

Every judged arm in this paper is scored against **one fixed opponent**: the unconstrained
`Llama-3.1-8B-Instruct`. That is deliberate --- a fixed opponent is what makes gains comparable
across arms --- but it means a reader cannot tell whether the reversal is a property of the two
mechanisms or of this particular opponent. It also entangles a disclosure the paper already makes:
judge~C is the same checkpoint as the opponent, a self-preference risk.

The five-judge panel changed the **instrument** and left the opponent fixed
(`onset_prediction_judge_panel.md`, `4` of `5`). This changes the **opponent** and holds the
instrument fixed. They are different questions and neither substitutes for the other.

## What is run

`Qwen2.5-14B-Instruct` --- a different family, roughly twice the size --- generates one completion
per prompt on the same `500` ordinary prompts, same decoding settings the committed opponent used
(temperature `1.0`, `max_new_tokens 200`, chat template, seed `1234`). Then
`analysis/order_averaged_h2h.py --baseline-dir <that run>` judges the same four arms against it,
both presentation orders, judge~B, `--seed 7717`.

**Nothing else changes.** The selection arm, the metered arm and both controls are the committed
generations, byte for byte. The only difference from the registered pass is which text sits on the
other side of every comparison.

## What is on record

Judge B, the committed opponent: `g_sel = +0.1045`, `g_met = +0.0400`, difference
**`+0.0645 [+0.0300, +0.0995]`**.

## Bands, committed before the run

**H1 --- the direction.** Read on the paired difference, within this pass.

| reading | band |
|---|---|
| REVERSAL HOLDS | `g_sel - g_met > 0` and its 95% interval excludes zero |
| UNRESOLVED | the interval contains zero |
| REVERSAL INVERTS | negative with the interval excluding zero |

**H2 --- the levels are expected to move, and are not read as a failure.** A stronger opponent
lowers every arm's win rate, so `g_sel` and `g_met` will both fall. **No level from this pass is
compared with any level from another** (caution (ap)); only the difference is.

**H3 --- the manuscript consequence, fixed now.**

- **REVERSAL HOLDS.** One sentence in Appendix~E: the comparison does not depend on the opponent's
  family or size. The headline stays the registered pass.
- **UNRESOLVED.** Reported as a limit on the judged claim: the difference is visible against this
  opponent and not against a stronger one, and the abstract's "four of five judges" gains "against
  one fixed opponent".
- **REVERSAL INVERTS.** Reported in the **main text**. An inversion against a different opponent
  would mean the judged comparison is about the opponent and not the mechanisms, and the paper
  would lean on the judge-free axis instead.

**We predict REVERSAL HOLDS but with a smaller difference**, because a stronger opponent compresses
both gains toward zero and the difference is a difference of gains.

## What may not be claimed

No certificate, leakage or `s(x)` number. No level quoted across passes. This arm says nothing
about the judge, which the panel arm covers, and nothing about the judge-free axis.

## Excluded alternatives

- Re-generating the selection or metered arms.
- Changing the judge, the seed, the prompt set or the decoding settings.
- Reporting this pass's levels beside the registered pass's.
- Picking whichever opponent agrees.

## Scoring log
