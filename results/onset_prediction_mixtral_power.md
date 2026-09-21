# Pre-registration: is Mixtral's exception a small effect or a wide interval? (feat-166)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists

Section 4 says the headline reversal *"replicates across judges, not across opponents"*: five
judges, `3.8`--`72`B in four families, re-scored byte-identical text, and **four of five** put the
paired difference clear of zero. The exception is the only family-clean frontier judge,
`Mixtral-8x7B-Instruct`, at **`+0.0090` `[-0.0355, +0.0530]`**.

That reading is UNRESOLVED in the registered sense --- the interval contains zero --- and the paper
reports it as such. But UNRESOLVED has two very different causes and the paper cannot currently
tell them apart. Either the effect under that judge is genuinely near zero, or `500` prompts is too
few to see a `+0.06` effect through that judge's noise. **Nothing about the judge can settle it:
`judge_batch` is greedy, so re-judging the same text gives the same answer exactly, and
order-averaging already removes the position lottery.** The only lever is more prompts.

This matters more than any other open reading in the paper. Every other judge agrees; if the one
clean frontier judge is measured at a tight zero, the honest summary of the headline changes from
"replicates across judges" to "replicates across judges except the largest clean one", and the
paper must say so.

## What runs

A **new, self-contained pass on a fresh prompt set**: `AlpacaEval-805`
(`data/bench/alpaca`, the factual slot so no `Complete the prefix:` header is prepended), which
feat-096 already established carries **no prompt-set effect** (A2). `805` prompts against the
committed `500` is a `1.61x` larger sample and a standard public benchmark rather than our own.

Nothing is compared across passes. Caution (ap) forbids setting a judged *level* from one sweep
against one from another, and this arm never does: every reading below is a **paired difference
within this pass**, and the committed `+0.0090` is quoted only as the prior it was registered
against.

| card | cell |
|---|---|
| 3 | anchor draws, `k=0`, `--trajectories-per-prompt 64` |
| 4 | metered decoder, `k=10`, one completion |
| 5 | the fixed opponent, unconstrained `Llama-3.1-8B-Instruct` |

Then `Qwen2.5-7B-Instruct` scores all `51{,}520` candidates once, and
`analysis/order_averaged_h2h.py` judges four arms in both presentation orders under **two** judges
--- judge B `Phi-3.5-mini` (the one on record) and `Mixtral-8x7B-Instruct` (the exception). The
other three judges are not re-run: they already agree, and spending four days re-confirming them
would buy nothing this arm is about.

## Gates, read before any band

- **G0 (the pipeline reproduces itself).** Judge B's paired difference on this pass must be
  positive with its interval clear of zero. Judge B is the arm on record at `+0.0645`
  `[+0.030, +0.0995]`; if it cannot reproduce the direction on a fresh prompt set, the pass is
  measuring something else and Mixtral's reading here is uninterpretable. **This is a direction
  gate, not a magnitude gate** --- a different prompt set may legitimately move the size.
- **G1 (the sample is bigger).** Judge B's interval half-width here must be **below** the `0.0348`
  on record. If a `1.61x` larger sample does not narrow the interval, the extra prompts are not
  buying power and B1 cannot be read as resolving anything.

## Bands, committed before the data

**B1 --- what Mixtral says with more prompts.**

| reading | band |
|---|---|
| **REVERSAL HOLDS** | difference `> 0` and interval excludes zero |
| **TIGHT ZERO** | interval contains zero **and** its half-width is `< 0.030`, i.e. the effect is bounded below the `+0.0645` on record |
| **STILL UNRESOLVED** | interval contains zero with half-width `>= 0.030` |
| **INVERTS** | difference `< 0` and interval excludes zero |

**B2 --- the paper's sentence, fixed in advance for each branch.**
REVERSAL HOLDS: the exception is retired and Section 4 says five of five, naming the sample size
that settled it. TIGHT ZERO: *"replicates under four judges and is bounded below `0.03` under the
one clean frontier judge"* --- the claim is weakened in the body, not in a footnote. STILL
UNRESOLVED: the current wording stands and this arm is reported as having failed to settle it.
INVERTS: the headline is withdrawn as judge-dependent.

**B3 --- reported whatever it says:** the paired `g_sel` and `g_met` under both judges, and
Mixtral's order-consistency rate on this pass against the `0.358`--`0.426` on record.

## Excluded in advance

- Pooling this pass with the committed one. The prompt sets are disjoint and nothing about pooling
  was registered; averaging them would be the post-hoc rescue this document exists to exclude.
- Quoting any level from this pass beside a level from another (caution (ap)).
- Adding judges after seeing B1, or dropping Mixtral if it reads INVERTS.
- Reading B1 if G0 fails.

## What we predict

**TIGHT ZERO.** The committed reading's point estimate is `+0.0090`, which is `0.20` interval
half-widths from zero --- caution (ap)'s rule says a reading that marginal does not survive a fresh
draw as anything but marginal, and the effect under this judge looks genuinely small rather than
merely noisy. If that is right the paper gets weaker and more accurate in the same edit.

## Compute

Host B. Generation about `3` hours across three cards, rewards about `10` minutes, judge B about
an hour, Mixtral on two cards about `6` hours.

## Scoring log
