# feat-156 --- CoTaEval news across seven anchors, and one disjoint re-draw

Everything above `## Scoring log` was written and committed **before any of these arms started**,
and nothing above that line is edited afterwards.

## Why

`feat-155` puts one anchor (Comma-7B) on CoTaEval's news utility half. One setup is exactly the
weakness the paper's own breadth work exists to fix: the judged claim rests on two anchors and the
appendix says so. If the CoTaEval result is to answer the Program Chairs' point about relying on
our own setups, it should not itself be a single setup.

This arm runs the **same** protocol at **seven anchors in three families**, and re-draws the
headline anchor with a disjoint seed. It is the breadth ladder the paper already reports on its own
corpora, re-run on the standard benchmark in a domain (news) the paper has never used.

## What runs --- one card each, host B, at the user's instruction to use all of them

Identical in everything but the anchor: CoTaEval `newsqa_indomain_utility` (`500` items),
`n in {1,2,4,8,16,32,64}`, pointwise reward `Qwen2.5-7B-Instruct` on the fixed template,
`--max-new 24`, `--batch-size 16`, SQuAD token-F1, `--seed 8801`.

| card | anchor | tag |
|---|---|---|
| 0 | Comma-7B (2T) | `_cta_news` (feat-155, already running) |
| 1 | Comma-7B (2T), **seed `5254`** | `_cta_s5254` |
| 2 | TinyComma-1.8B | `_cta_tc18b` |
| 3 | KL3M-3.7B | `_cta_kl3m37b` |
| 4 | Pleias-3B | `_cta_pleias3b` |
| 5 | Comma-7B (1T) | `_cta_comma1t` |
| 6 | KL3M-1.7B | `_cta_kl3m17b` |
| 7 | Pleias-1.2B | `_cta_pleias12b` |

`--seed` drives `torch.manual_seed` as well as the bootstrap, so card 1 is a genuine disjoint draw
and not merely a different resample. **Batch size is held at `16` everywhere**, because batch size
is part of the seed and a shift at a rate-valued quantity (cautions (u), (v)); an arm that changed
the seed AND the batch size would not be the comparison it was registered as.

## Gates, per anchor, read before that anchor's band

Identical to `feat-155`, applied **per anchor** and never pooled:

* **G1** the anchor's own `n=1` F1 is **`>= 0.10`**. Below it the anchor cannot read the article and
  no selection number over it means anything; that anchor is reported as UNUSABLE ON THIS TASK and
  its band is not computed. This is expected to fire at the small anchors and is not a failure of
  the mechanism --- it is the capability floor the paper already concedes.
* **G2** the unconstrained risky model beats that anchor's `n=1`. If not, the arm is INVALID.
* **G3** the `n=1` empty fraction is reported, never gated against a prior (caution (v)).
* **G4** all `500` items at every `n`.

## Bands

**Per anchor**, the paired `F1(64) - F1(1)` within that anchor's own pass:
`> 0` with the interval excluding `0` is **CLIMBS**; an interval containing `0` is **NO EFFECT**;
`< 0` excluding `0` is **TURNS OVER**. Below **`2.0`** interval half-widths the reading is
**MARGINAL** and is not promoted without a seed replication (caution (ap)).

**The ladder claim.** If CoTaEval news behaves like our own corpora, **at least two anchors climb
with intervals excluding zero**, and the anchors that climb are the capable ones. REFUTED if fewer
than two climb, in which case the paper reports that the judge-free result does not transfer to
this benchmark and says so in Limitations.

**The replication (card 1) is the test of our own rule, out of sample.** Caution (ap) holds that a
paired difference reproduces above about `2.0` interval half-widths and not below. The prediction
is therefore **conditional and fixed now, before either number exists**: if feat-155's reading is
at or above `2.0` half-widths the re-draw **keeps its verdict**; if below, it **need not**, and a
changed verdict there is evidence for the rule rather than against the arm. Recording it this way
is what stops the rule being fitted after the fact.

## What may not be claimed

* No F1 level from one anchor's pass compared against another pass's level (caution (ap)); only
  each anchor's own paired difference, and the count of anchors that climb.
* No anchor's reading pooled with the replication's (caution (ap)).
* No leakage, certificate or `s(x)` number: this half is utility only.
* An anchor failing G1 is **not** evidence about selection anchoring. It is evidence the anchor
  cannot do reading comprehension, which the paper already states as the mechanism's binding
  constraint.

## Compute

Seven arms of `500 x 64` at `1.2`--`7`B, one card each on eight idle H100s, comparable arms having
taken `2.5`--`3.5` h per `32,000` trajectories. **No single arm is near the `24`-gpu-hour
escalation threshold**; the aggregate is large only because the cards are parallel and idle.

## Scoring log

All seven arms plus the feat-155 primary finished on host B, 2026-09-20, one card each, no errors.
`analysis/score_cotaeval.py` -> `results/cotaeval_scoring.csv`.

### G1 split the ladder in half, exactly as registered

| anchor | `n=1` F1 | G1 (`>= 0.10`) |
|---|---|---|
| Comma-7B (2T) | `0.4291` | PASS |
| Comma-7B (2T), seed `5254` | `0.4562` | PASS |
| Comma-7B (1T) | `0.3496` | PASS |
| TinyComma-1.8B | `0.1669` | PASS |
| Pleias-1.2B | `0.0879` | **FAIL** |
| KL3M-3.7B | `0.0499` | **FAIL** |
| KL3M-1.7B | `0.0294` | **FAIL** |
| Pleias-3B | `0.0027` | **FAIL** |

Four anchors cannot read a news article at all. **No band was computed for them** --- the
registration forbids it and the scorer refuses. This was registered in advance as the expected
outcome and as the **capability floor the paper already concedes**, not as evidence about the
mechanism, and it is reported that way. G2 passed everywhere.

### The ladder claim is REFUTED

The band: *"at least two anchors climb with intervals excluding zero"*. Under the registered
pointwise reward, **zero anchors climb. All four usable anchors TURN OVER:**

| anchor | `F1(1)` | `F1(64)` | paired gain | hw | verdict |
|---|---|---|---|---|---|
| Comma-7B (2T) | `0.4291` | `0.2455` | `-0.1836 [-0.2249, -0.1412]` | `4.39` | TURNS OVER |
| Comma-7B (2T) seed `5254` | `0.4562` | `0.2228` | `-0.2334 [-0.2738, -0.1922]` | `5.72` | TURNS OVER |
| Comma-7B (1T) | `0.3496` | `0.2372` | `-0.1124 [-0.1543, -0.0711]` | `2.70` | TURNS OVER |
| TinyComma-1.8B | `0.1669` | `0.1075` | `-0.0594 [-0.0921, -0.0274]` | `1.84` | TURNS OVER (marginal) |

Three of the four are far outside the marginality band. The registered consequence applies: *"the
paper reports that the judge-free result does not transfer to this benchmark and says so in
Limitations"*, and feat-155's TURNS OVER branch puts the finding in the **main text**.

### Majority vote, beside it

| anchor | paired gain | verdict |
|---|---|---|
| Comma-7B (2T) | `+0.0284 [-0.0012, +0.0576]` | NO EFFECT (marginal) |
| Comma-7B (2T) seed `5254` | `-0.0011 [-0.0306, +0.0298]` | NO EFFECT |
| Comma-7B (1T) | `+0.0230 [-0.0110, +0.0568]` | NO EFFECT (marginal) |
| TinyComma-1.8B | `-0.0351 [-0.0603, -0.0100]` | TURNS OVER (marginal) |

The rule with no scorer to overoptimise against does not collapse --- but it does not climb either.
**On this benchmark neither rule buys utility.**

### The replication answers the rule it was registered against

feat-155's reading sits at `4.39` half-widths, **above** the `2.0` boundary, so the conditional
prediction fixed before either number existed was that the disjoint draw **keeps its verdict**. It
does: `-0.2334` against `-0.1836`, TURNS OVER both times, `5.72` and `4.39` half-widths. Caution
(ap)'s rule is tested out of sample here on a NEGATIVE reading for the first time and holds.

The two levels differ by `0.05`, which is **not** read as instability: the arms judge disjoint
candidate sets and no level is compared across passes (caution (ap)). Only the verdict and the
paired difference within each pass are.

### What may not be concluded

* Not a refutation of the certificate, which does not depend on the served text being good.
* Not a refutation of GSM8K majority vote (`0.320 -> 0.546`): different task, different rule.
* The four G1 failures say nothing about selection anchoring. They say a legal-domain or small
  anchor cannot do news reading comprehension, which is the constraint the paper already names.
* The four arms are **not pooled** with each other or with the replication.
