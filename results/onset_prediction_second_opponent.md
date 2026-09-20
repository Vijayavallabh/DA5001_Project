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

Run 2026-09-20, one H100. Generation `14:02--15:14` (exit 0), judge pass `15:14--` (exit 0).

```
bash scripts/run_second_opponent.sh 0
```

`results/order_averaged_h2h__opp2.csv`.

| quantity | value | 95% interval | reading |
|---|---|---|---|
| D1 selection gain, order-averaged | `+0.0540` | `[+0.0320, +0.0765]` | SURVIVES |
| D2 metered gain, order-averaged | `+0.0605` | `[+0.0385, +0.0820]` | SURVIVES |
| **D3 difference of gains, paired** | **`-0.0065`** | **`[-0.0385, +0.0255]`** | see below |

### H1: UNRESOLVED

The registered taxonomy is three-way and keys on the interval: HOLDS is `> 0` **and** the interval
excludes zero; UNRESOLVED is *the interval contains zero*; INVERTS is negative **and** the interval
excludes zero. The interval is `[-0.0385, +0.0255]`, which contains zero, so the registered reading
is **UNRESOLVED**.

**The point estimate is negative and that is not hidden behind the label.** It is `-0.0065`, which
is `0.20` interval half-widths from zero --- the difference is not merely unresolved, it is
indistinguishable from nothing against this opponent.

**The CSV's own `reading` column says `REVERSAL REFUTED` and that string is not the registered
reading.** `analysis/order_averaged_h2h.py:218` labels *any* negative point estimate REFUTED
without consulting the interval, while requiring a positive one to clear the interval before it
says CONFIRMED --- an asymmetric rule, and not the one committed here. Reported so the softer word
cannot be mistaken for a choice made after seeing the number; caution (ag), a label a script writes
is not a measurement. The asymmetry is logged as a defect in that script, not repaired in this arm.

### H2: the prediction was unfalsifiable as written, which is a defect in our own specification

H2 predicted that both levels would **fall** against a stronger opponent, and the same paragraph
forbade comparing any level from this pass with any level from another (caution (ap)). Those cannot
both be honoured: the prediction is about a cross-pass level comparison that the arm's own rules
exclude. For the record and without drawing a conclusion from it, `g_met` did not move in the
predicted direction. No inference is taken from that, because the comparison is exactly the one
caution (ap) says is invalid --- a different opponent changes every pair's content, so the two
passes are not comparable at the level. **Record the defect rather than quietly repairing it**
(caution (w)): a future opponent-transfer arm should predict only the paired difference, which is
within-pass and was correctly specified here.

### H3: the registered consequence, applied

The UNRESOLVED branch was fixed before the run and is applied verbatim: *reported as a limit on the
judged claim --- the difference is visible against this opponent and not against a stronger one,
and the abstract's ``four of five judges'' gains ``against one fixed opponent''.* Both changes are
made.

### What this arm does and does not overturn

**Does not.** Both mechanisms still beat their own controls against this opponent, each interval
clear of zero (D1 `+0.054`, D2 `+0.0605`). Nothing about the certificate changes: `log n` is a
statement about the served law and no judge enters it. The judge-free axis is untouched --- it has
no opponent.

**Does.** The judged head-to-head between the two mechanisms is **opponent-dependent**, and the
paper may no longer present it as a property of the mechanisms alone. The five-judge panel changed
the instrument and held the opponent fixed; this changed the opponent and held the instrument
fixed, and it is the second that does not survive.

### One observation, offered as consistency and not as a prediction

The registered pass's D3 sits `1.86` interval half-widths from zero. Caution (ap)'s own rule ---
written about a different perturbation, re-drawing candidates --- is that a paired difference
reproduces at about `2.1` half-widths and is unsafe below it, with `1.71` the measured failure.
`1.86` is inside that unsafe band. The perturbation here is not the one that rule was measured on,
so this is noted as consistent rather than as something the rule foresaw.

Order consistency this pass: `sel_n64` `0.550`, `sel_n1` `0.628`, `anchor_k0` `0.626` (NOISY), and
`metered_k10` **`0.482`** (UNUSABLE, below chance agreement between presentation orders). Order
averaging is what makes the gains readable at all here; the single-order D3 is `+0.007`.
