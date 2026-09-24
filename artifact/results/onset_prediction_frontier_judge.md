# Pre-registration: does the headline survive a frontier-class judge?

Committed **before either judge is run**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

Every referee report raises the judge. The headline reversal --- selection `+0.1045` against the
metered decoder `+0.0400`, an order-averaged paired difference of `+0.0645 [+0.030, +0.0995]` ---
is carried by **judge B, `Phi-3.5-mini-instruct`, a 3.8B model**. One report asks the question in
as many words: *"Does the +0.1045 vs +0.0400 head-to-head replicate under judge A (Qwen2.5-7B) and
a frontier-class judge? Given the 0.04--0.07 floor you establish, justify the headline's reliance
on a 3.8B judge."* Another asks for the same thing as a condition of raising its score.

The paper cannot answer it today. Judge A and judge C are both 7--8B, judge C is the opponent's own
checkpoint, and no judge above 8B has ever scored this comparison.

**This arm changes only the instrument.** It re-scores the *same committed generations* --- byte
for byte, the texts behind `results/order_averaged_h2h.csv` --- with two larger judges. Nothing is
regenerated, no reward is recomputed, and no model that produced any text is involved in judging
it. Under a greedy judge, order-averaged scoring is a deterministic function of the text
(`results/n128_order_averaged_note.md`), so a disagreement here is the instrument and nothing else.

## Why two judges, and why these two

A single frontier judge would answer "does a bigger model agree" and leave "does a bigger model of
*this family* agree" unanswered. The paper's deployable scorer is `Qwen2.5-7B-Instruct`, so a Qwen
judge shares a family with the thing that did the selecting --- a self-preference risk of exactly
the kind the paper already discloses for judge C.

| judge | model | size | family shared with |
|---|---|---|---|
| **D** | `Qwen/Qwen2.5-72B-Instruct` | 72B | the **scorer** (Qwen2.5-7B-Instruct) |
| **E** | `mistralai/Mixtral-8x7B-Instruct-v0.1` | 47B (8x7B MoE) | **nothing** in this paper |

Judge E is the clean one and judge D is the powerful one. Both are reported, separately, whatever
they say, and neither replaces judge B in the headline: the registered headline stays judge B
unless H3 below fires.

## What is run

```
analysis/order_averaged_h2h.py --judge <model> --device-map auto --tag _<judge> --out results
```

on `output/xfer/sel_anchor64` (selection), `output/xfer/conc_all` (metered, `k=10`),
`output/sweep_plain` (the `k=0` anchor control and the `k=-1` opponent), rewards
`results/selection_rewards64.csv`, `--n 64`, `--seed 7717`. Four arms --- `sel_n64`, `sel_n1`,
`metered_k10`, `anchor_k0` --- each judged against the same fixed opponent in **both** presentation
orders, exactly as the committed pass did.

`--device-map auto` is new and is the only code change: both judges exceed one card. It changes
placement, not arithmetic.

## What is already on record, and is the thing to reproduce

From `results/order_averaged_h2h.csv`, judge B, order-averaged:

| quantity | value |
|---|---|
| selection gain, `g_sel` | `+0.1045` |
| metered gain, `g_met` | `+0.0400` |
| **paired difference `g_sel - g_met`** | **`+0.0645 [+0.030, +0.0995]`** |

The **difference** is the claim; the levels are not, and are not compared across judges.

## Bands, committed before the run

**H1 --- the direction. Does the reversal hold at all?** Read on each judge separately.

| reading | band |
|---|---|
| REVERSAL HOLDS | `g_sel - g_met > 0` and its 95% interval excludes zero |
| UNRESOLVED | the interval contains zero |
| REVERSAL INVERTS | `g_sel - g_met < 0` and the interval excludes zero |

**H2 --- the magnitude. Is it the same size?** The committed difference is `+0.0645`. The paper's
own measured cross-pass floor on a judged gain is `0.04`, but that floor is about *generation*
runs; this is the same text, so the floor does not apply and a tighter band is fair.

| reading | band |
|---|---|
| AGREES | the new difference lies within `0.03` of `+0.0645` |
| SAME SIGN, DIFFERENT SIZE | same sign, differs by more than `0.03` |
| DISAGREES | opposite sign, or interval excludes `+0.0645` on the wrong side |

**H3 --- the manuscript consequence, fixed now for every combination.**

- **Both judges REVERSAL HOLDS.** Section 4.3 gains one sentence naming both judges, their sizes
  and their family relation to the scorer, and the Limitations sentence on judge size is **struck**.
  The headline stays judge B; what changes is that it is no longer the only instrument that has
  seen the comparison.
- **Judge E (clean family) holds, judge D does not.** Report both and keep the headline. The
  paper says plainly that the judge sharing the scorer's family is the one that disagreed, which
  is evidence *against* self-preference inflating our number, not for it.
- **Judge D holds, judge E does not.** This is the damaging combination and it is reported as
  such: the reversal would then be visible only to judges sharing the scorer's family, and
  Section 4.3 must say so and the abstract must carry the qualifier "under judges related to the
  scorer".
- **Neither holds.** The head-to-head is an artefact of a 3.8B judge. The abstract's `+0.1045`
  against `+0.0400` comes out, the paper reports the judge-free axes (GSM8K, TriviaQA) as the
  utility evidence, and the contribution narrows to the certificate plus the dichotomy. **We commit
  to this now, before seeing a number.**
- **Any judge INVERTS.** Same as "neither holds", plus the inversion is stated in the abstract.

**H4 --- what may not be claimed.** No level from this arm is quoted beside a level from any other
pass or judge (caution (ap)). No certificate, leakage or `s(x)` number comes from here. The
single-order numbers this script also writes are **not** read for H1--H3; they are recorded only
to show the order-averaging correction on a third and fourth instrument.

## Excluded alternatives

- Picking whichever judge agrees and reporting only that one.
- Re-judging with a different prompt, template, temperature or max-token budget than the committed
  pass used. `judge_batch` is greedy and its template is fixed; only the model changes.
- Reading H2 as a failure if H1 holds --- a size disagreement on a new instrument is expected and
  is not evidence against the direction.
- Regenerating any text. If a generation directory is missing, the arm does not run.
- Treating a 72B judge as ground truth. It is a larger instrument, not a correct one.

## Scoring log

## Scoring, 2026-09-20

Run 2026-09-20 on the DGX, `analysis/order_averaged_h2h.py --device-map auto`, both judges on the
same committed generations, `--seed 7717`, `--n 64`. Outputs `results/order_averaged_h2h__qwen72b.csv`
and `results/order_averaged_h2h__mixtral.csv` with their per-prompt twins.

### The numbers

| judge | family | `g_sel` | `g_met` | **`g_sel - g_met`** | order consistency |
|---|---|---|---|---|---|
| B `Phi-3.5-mini` 3.8B (on record) | clean | `+0.1045` | `+0.0400` | **`+0.0645 [+0.0300, +0.0995]`** | `0.244`--`0.350` |
| **D** `Qwen2.5-72B-Instruct` | **scorer's** | `+0.1950` | `+0.1330` | **`+0.0620 [+0.0145, +0.1100]`** | `0.742`--`0.776` |
| **E** `Mixtral-8x7B-Instruct` | clean | `+0.1060` | `+0.0970` | **`+0.0090 [-0.0355, +0.0530]`** | `0.358`--`0.426` |

### H1 --- the direction

- **Judge D: REVERSAL HOLDS.** `+0.0620`, interval excludes zero.
- **Judge E: UNRESOLVED.** `+0.0090`, interval contains zero.

### H2 --- the magnitude

- **Judge D: AGREES.** `|+0.0620 - (+0.0645)| = 0.0025`, inside the committed `0.03`.
- **Judge E: DISAGREES.** Same sign, but the band's second clause fires: its interval tops out at
  `+0.0530` and so **excludes `+0.0645` on the low side**. Scored under the band as written, not
  under the milder `SAME SIGN, DIFFERENT SIZE` that the point estimate alone would have given.

### Both instruments are healthy, and that was checked before anything was concluded

Neither judge is degenerate. Judge E's mass sits on ties (`163`--`195` of `500` order-averaged
verdicts at exactly `0.5`); judge D's is bimodal at `0`/`1` (`178`+`197` on `sel_n64`). Both produce
the full five-point order-averaged support. So judge E is *indecisive*, not broken --- and
indecision cannot be dismissed as noise here, because **judge B is the least self-consistent
instrument of the three (`0.244`--`0.350`) and it resolves the difference**. Low consistency does
not predict failure to resolve; nothing we measured does.

Cross-judge item-level agreement is weak in both directions: the per-prompt difference correlates
at `r = +0.284` between D and E, with sign agreement on `302/500` prompts. The judges agree on the
aggregate direction of `g_sel` far better than they agree on any individual prompt.

### H3 --- the branch that fired, and a defect in this pre-registration

The combination is **"judge D holds, judge E does not"** --- the branch this document names as
*"the damaging combination"*. It is reported as such, and the consequence is applied.

**But the branch's stated reason is false, and it is falsified by data that predates this arm.**
It reads *"the reversal would then be visible only to judges sharing the scorer's family"*. Judge B
is `Phi-3.5-mini-instruct`, a Microsoft model with no family relation to `Qwen2.5-7B-Instruct`, and
it resolves the difference at `+0.0645`. Family therefore does **not** order the outcome: one clean
judge resolves, one clean judge does not, and the scorer-family judge resolves. Neither does size
(`3.8B` resolves, `47B` does not, `72B` resolves) nor self-consistency (see above).

Writing the registered qualifier *"under judges related to the scorer"* into the abstract would
put a **false sentence** in the paper. The defect is that this branch labelled its outcome by a
*hypothesised cause* rather than by the *observable*, and the hypothesis is refuted. Per caution
(w) the specification's defect is recorded rather than silently repaired, and the deviation is
stated here in full:

- **Kept:** the branch's substance --- this is damaging, it is reported as damaging, the abstract
  carries a qualifier, and Section 4.3 states the disagreement.
- **Changed:** the qualifier's *content*, from the false *"under judges related to the scorer"* to
  the measured *two of three judges resolve the difference; one does not*.
- **Direction of the deviation, stated because it matters:** the registered wording is the
  **more** damaging of the two, so this correction makes the paper look better, which is the
  direction in which a post-hoc change is least trustworthy. It is made only because the registered
  wording is factually untrue, and the truthful replacement is still a demotion of the headline.

### The consequence actually applied to the manuscript

What survives **all three** judges, and is what the paper may assert:

1. Selection's gain over its own control is positive and its interval excludes zero under every
   judge (`+0.1045`, `+0.1950`, `+0.1060`).
2. The metered decoder's gain is likewise positive under every judge.
3. The **certificate** ratio is untouched by any of this: `3.175` nats against a measured `171.3`
   is arithmetic and measurement, not judgment.

What does **not** survive is the strict inequality. The headline is therefore demoted from
*selection gains **more** than the meter* to **selection matches or exceeds the meter at one
fifty-fourth of the divergence**, with all three judges reported. The `+0.1045` against `+0.0400`
levels come out of the abstract: they are judge B's, they move by a factor of three across judges
(`g_met`: `+0.0400`, `+0.0970`, `+0.1330`), and caution (ap) forbids quoting a judged level across
passes in any case.

**Not done, and deliberately:** the three judges are not pooled. Pooling was not registered,
caution (ap) forbids averaging a reading with its own failed replication, and a mean over three
instruments that disagree at `r = +0.284` per item would manufacture a precision none of them has.
