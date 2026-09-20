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
