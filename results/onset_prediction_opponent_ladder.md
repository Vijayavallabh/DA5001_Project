# Pre-registration: is the judged head-to-head a function of the opponent's STRENGTH?

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists

`onset_prediction_second_opponent.md` (feat-153) swapped the committed opponent
`Llama-3.1-8B-Instruct` for `Qwen2.5-14B-Instruct` and the paired difference `g_sel - g_met` went
from `+0.0645 [+0.0300, +0.0995]` (REVERSAL CONFIRMED) to `-0.0065 [-0.0385, +0.0255]`
(REVERSAL UNRESOLVED). The appendix reports that honestly and reports it as far as it goes: *"the
judged head-to-head between the two mechanisms is opponent-dependent"*, and the paper moved its
weight onto the certificate and the judge-free axis in consequence.

**That sentence is built on two points, and the swap moved two things at once** --- family and size
--- so nothing about it is identified. A reader is entitled to ask whether the paper's headline
comparison is an artefact of one particular opponent, and two points cannot answer that.

Measured now, before anything is run, from the two per-prompt files already on disk: the committed
opponent beats the anchor control on **`0.555`** of prompts and `Qwen2.5-14B-Instruct` on
**`0.850`**. **The "second opponent" was not a perturbation.** It is a `0.295` move on a bounded
scale, and neither the registration nor the appendix says so. A difference of gains has very little
room against an opponent that wins `85%` of the time, so **compression** is a live and untested
explanation for the whole result, and it is a far less damaging one than "the comparison is about
the opponent".

## The axis, and why it may be compared across passes

    opponent strength := 1 - mean(u_anchor_k0)

`u_anchor_k0` is the anchor control's **order-averaged** win rate against that opponent. This is a
judged LEVEL, and caution (ap) forbids comparing a level across passes, so the exemption is argued
rather than assumed:

1. the reference text is `output/sweep_plain`, **byte-identical in every pass**;
2. order averaging judges both presentation orders, so it consumes nothing from the rng --- caution
   (ap)'s own measurement is that it is then a deterministic function of the text under a greedy
   judge, reproducing a committed pass to four decimals where single-order moved `0.066`;
3. the judge, its template, its seed and the prompt set are fixed.

So the number differs between passes **only through the opponent's text**, which is the variable
under study. That is the one level comparison these rules permit, and the reason does not
generalise: a *gain* measured against a different opponent is not comparable, because the gain's
own control is also judged against that opponent.

## What is run

Three more opponents, **within one family so that size is the only thing that moves**:
`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`. With the
two on record this gives five opponents, four of them one family spanning `0.5`B to `14`B.

Each generates one completion per prompt on the same `500` ordinary prompts at the settings the
committed opponent used (temperature `1.0`, `max_new_tokens 200`, chat template, seed `1234`), and
then `analysis/order_averaged_h2h.py --baseline-dir <that run>` judges the **same four committed
arms** against it under judge~B, `--seed 7717`, both presentation orders.

**Only `--baseline-dir` changes.** The selection arm, the metered arm, both controls and the reward
cache are the committed generations byte for byte; nothing is re-drawn and nothing is re-scored.

## The entanglement, disclosed in advance, and which way it cuts

The pointwise scorer is `Qwen2.5-7B-Instruct`, so these three opponents share its family. Judge~B
is `Phi-3.5-mini` and is family-clean with respect to both. If a shared family gives the scorer a
stylistic preference the opponents also express, selection's picks look **more like** the opponent
and the judge is pushed toward a draw --- which depresses `g_sel`, lowers `D3`, and pushes **away
from** REVERSAL CONFIRMED. **The entanglement is conservative for the hypothesis H1 tests**, so a
confirmation is not explained by it. A refutation partly could be, and that is stated here so it
cannot be argued afterwards.

## Bands, committed before the run

### H1 --- primary, and on new data only

If the result is driven by opponent strength, then **every new opponent measuring strength below
the committed opponent's `0.555` must read `REVERSAL CONFIRMED`** --- `D3 > 0` with its 95%
interval excluding zero.

| reading | band |
|---|---|
| **STRENGTH SUPPORTED** | every new opponent with strength `< 0.555` reads REVERSAL CONFIRMED |
| **STRENGTH REFUTED** | at least one opponent with strength `< 0.555` reads UNRESOLVED or REFUTED |
| **NOT TESTED** | no new opponent measures below `0.555` |

A new opponent measuring **above** `0.555` is not a test of this prediction, is excluded from H1,
and is reported on its own line. The threshold `0.555` is the committed opponent's measured
strength and is fixed by data already on disk, not chosen here.

### H2 --- the ceiling at the other end, registered so it cannot be discovered

A difference of gains is compressed at **both** ends. Against an opponent the anchor control
already beats, every arm's win rate runs into the same ceiling and `D3` has no room either.

| reading | band |
|---|---|
| **COMPRESSION AT BOTH ENDS** | an opponent with strength `< 0.30` reads UNRESOLVED while one between `0.30` and `0.555` reads CONFIRMED |

If this fires it is reported **instead of** STRENGTH REFUTED for that opponent, because a floor
artefact is not evidence against the strength account --- and it would say something sharper and
worse: that the committed opponent sits near the only place on the scale where the instrument has
room to show anything at all.

### H3 --- exploratory, and labelled one

Spearman of `D3` against measured strength over all five opponents, with exact two-sided
permutation `p`. **Two of the five points are already seen**, so this is descriptive and is
reported as exploratory. It is not a band and nothing is concluded from it alone.

### H4 --- the manuscript consequence, fixed now

- **STRENGTH SUPPORTED.** The concession in `app:h2hrepeat` becomes a measurement instead of an
  anecdote: the judged difference is present against opponents up to strength `~0.555` and
  compresses against stronger ones, with every opponent's strength printed so a reader can place
  them. The abstract's *"against one fixed opponent"* qualifier **stays** --- a compression
  account explains the disappearance, it does not restore the claim at `0.850`.
- **STRENGTH REFUTED.** The concession stands and hardens: the difference is not a function of
  opponent strength, so the opponent-dependence is unexplained, and that sentence moves from the
  appendix into the main text's limitations.
- **COMPRESSION AT BOTH ENDS.** Reported in the appendix as a property of the *instrument*: the
  judged head-to-head can only resolve a difference in a window of opponent strength, and the
  paper's is measured inside it. This is a reason to prefer the judge-free axis and is written as
  one.
- **NOT TESTED.** The arm is reported as having failed to place a point below `0.555` and no
  consequence is drawn. The models were chosen expecting them to be weaker; if a `0.5`B instruct
  model beats the anchor control more often than `Llama-3.1-8B-Instruct` does, that is itself
  worth a sentence and nothing more.

**We predict STRENGTH SUPPORTED**, because `0.555 -> 0.850` is a large move on a bounded scale and
a difference of gains has to shrink somewhere. We note in advance that predicting it does not make
H2 less likely, and H2 is the reading we would least like.

## Gates, read in this order, before any band

- **G1 --- the corpus and the settings.** Each new opponent run must hold `500` completions over
  the same prompt ids as the committed pass, at temperature `1.0`, `max_new_tokens 200`, chat
  template, seed `1234`. Read off the run, not off this document.
- **G2 --- only the opponent changed.** The `--sel-dir`, `--metered-dir`, `--anchor-dir`,
  `--rewards`, `--judge`, `--n`, `--k` and `--seed` arguments must be identical to the committed
  pass's. An arm that changed anything else is INVALID, not failed (caution (w)).
- **G3 --- an opponent that is degenerate is not a weak opponent.** Fewer than `10%` empty
  completions, and a mean completion length within a factor of `3` of the committed opponent's.
  **Both halves are derived from `output/sweep_plain` by the scorer** (`0.0000` empty, `154.08`
  words) and never typed. A run that fails G3 is excluded from H1 and H2 and reported as excluded:
  its low strength would be a parsing or truncation artefact read as capability, which is caution
  (au)'s shape.

G3's reference is derived by the scorer at run time. The numbers quoted above are what it prints
today and are recorded so that a change in them is visible, not so that they can be typed in.

## What may not be claimed

- No certificate, leakage or `s(x)` number. Proposition~1 does not involve a judge or an opponent.
- No judged **level** quoted across passes, with the single exception argued above, which is a
  level against a byte-identical reference and is used only as the ordering variable.
- Nothing about the judge --- `onset_prediction_judge_panel.md` is that question, and this arm
  holds the judge fixed exactly as that one holds the opponent fixed.
- Nothing about the workload. Every pass here is the same `500` ordinary prompts.
- No pooling of a new opponent with the two on record into a single number.

## Excluded alternatives

- Re-generating the selection arm, the metered arm or either control.
- Changing the judge, the seed, the prompt set, `n`, `k` or the decoding settings.
- Adding an opponent after seeing another opponent's result, or dropping one that has been run.
  The ladder is fixed at these three and is the whole ladder.
- Using `Phi-3.5-mini`, `Qwen2.5-14B`, `Qwen2.5-72B`, `Mixtral-8x7B` or `gemma-2-27b-it` as an
  opponent: all five are judges on this paper's panel, and an opponent that is also an instrument
  is not a clean point.

## Scoring log
