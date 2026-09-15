# Pre-registration: does the headline reversal survive order-averaging on BOTH arms?

Committed **before the arm runs**, as every arm in this line has been. Nothing above the
`## Scoring log` rule may be edited after the run; the scoring log is appended underneath.

## Why this arm exists

A reviewer of the v8 draft made the sharpest experimental objection the paper has received, and it
is correct as stated:

> The paper's own check shows the judge produces mutually consistent verdicts on only 29.2% of
> items. The headline reversal (+0.142 vs +0.072) is a factor-of-two gap between point estimates of
> ~0.07-0.14 measured with an instrument whose cross-pass floor the authors themselves put at
> +/-0.04. Worse, order-averaging cuts the marquee judge-A-selects/judge-B-scores gain from +0.081
> to +0.0405 -- a halving the text reports and then largely ignores in the abstract.

Everything in that paragraph is ours. `results/judge_consistency.csv` reads C1 = 0.292 UNUSABLE,
C2 = 0.127 STRONG, and C3 = 0.0405 against a recorded 0.081. The paper then quotes, as its central
empirical sentence, a comparison of two gains **neither of which was order-averaged**.

The comparison is not thereby wrong: both arms were judged by the same instrument in one random
order per pair, so position bias is a common artefact and the arms are like-for-like. But the
paper's own instrument check establishes that a single-order gain is roughly twice what survives
order-averaging, and it has never been checked whether the **difference** survives. That difference
is what the abstract claims. Nobody has computed it, including us.

## The estimand nobody has computed

Let `g_S` be selection anchoring's gain over its own anchor-alone control and `g_M` the metered
decoder's gain over the same control, both with **position removed by construction** rather than in
expectation -- each item judged in both presentation orders and its two utilities averaged. The
quantity the abstract asserts is

    D = g_S - g_M ,  paired over the shared prompts.

## Design

No generation. A second and third scoring pass over text already on disk, by judge B
(`microsoft/Phi-3.5-mini-instruct`), template and 4-token greedy decode unchanged from every other
judged arm in this paper.

* **Arm S** -- selection at `n = 64` with the pointwise Qwen reward, candidates
  `output/phase5/sel_anchor64`, picks reproduced from the cached `results/selection_rewards64.csv`
  so the served text is bit-identical to the arm that produced `+0.142`. Control: `n = 1`, the
  first draw, exactly as `selection_scaling.py` defines it.
* **Arm M** -- the metered decoder at `k = 10`, the arm that produced `+0.072`
  (`results/judge_separation_v6_judge2.csv`: KL k=10 utility `0.522` against `anchor only`
  `0.4505`). Control: the same anchor-alone text.
* Every item in **both** orders. Utility `win = 1, tie = 0.5, loss = 0`, order-averaged per item.
* Intervals: bootstrap over prompts, 10,000 resamples, paired wherever the two arms share a prompt
  id. The pairing is reported, not assumed.

## Bands, committed before the run

* **D1 -- selection's order-averaged gain at n = 64.**
  `SURVIVES` if its 95% CI excludes 0; `DISSOLVES` if it contains 0.
  Committed expectation: it survives at roughly half the single-order value, in `[0.05, 0.11]`.
* **D2 -- the metered decoder's order-averaged gain at k = 10.** Same test, same reasoning.
  Committed expectation: `[0.02, 0.06]`.
* **D3 -- the headline. `D = g_S - g_M`, paired.**
  * `REVERSAL CONFIRMED` -- `D > 0` and its 95% CI excludes 0.
  * `REVERSAL UNRESOLVED` -- `D > 0` and its CI contains 0.
  * `REVERSAL REFUTED` -- `D <= 0`.
* **D4 -- the divergence ratio.** `3.175` nats against `171.3` is an accounting fact and is not on
  trial here. It is named so that what D3 can and cannot cost the paper is fixed in advance.

## What each reading costs, committed in advance

* `REVERSAL CONFIRMED` -- the abstract keeps its claim and gains the order-averaged numbers, which
  is strictly better evidence than it has now.
* **`REVERSAL UNRESOLVED` -- the abstract and introduction stop saying "twice the utility". They
  state that selection reaches the metered decoder's judged utility at a fifty-fourth of its
  divergence, and that the two mechanisms' judged utilities do not separate at this instrument's
  resolution. The Limitations paragraph says so in those words.**
* `REVERSAL REFUTED` -- the constructive claim is restated as parity at lower divergence under a
  stronger order, and the paper says the judged comparison went against it.

I predict **D3 = REVERSAL UNRESOLVED**: the one order-averaging on record moved a gain by a factor
of `0.50`, and if both arms move by that factor then `D ~ 0.035` against a cross-pass floor of
`~0.04`. That prediction costs the paper its central empirical sentence, which is the point of
writing it down before the run rather than after.

## Alternatives excluded in advance

* The secondary "consistent items only" statistic **will not** be promoted to primary if the
  primary dissolves. It conditions on an outcome, and `judge_consistency.csv` already marks it
  secondary for that reason.
* Judge C **will not** be substituted to rescue a null.
* No other `n`, no other `k`, and no other control will be tried to find a cell that separates.
* If Arm M's per-item generations prove unrecoverable from disk, D3 is reported as
  `NOT MEASURABLE`, the abstract is weakened exactly as under `UNRESOLVED`, and the reason is
  stated. A missing artefact is not permitted to leave the claim standing by default.

## Scoring log

## Scoring, 2026-09-14

**Run.** `output/logs/order_averaged_h2h.log`, GPU 1, 2026-09-14. 4,000 judged pairs (four arms
x 500 prompts x both orders), judge `microsoft/Phi-3.5-mini-instruct`, no generation.

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_averaged_h2h.py --out results
```

| band | value | 95% CI | reading |
|---|---|---|---|
| D1 selection gain, n=64, order-averaged | `+0.1045` | `[+0.082, +0.128]` | **SURVIVES** |
| D2 metered gain, k=10, order-averaged | `+0.0400` | `[+0.014, +0.0655]` | **SURVIVES** |
| D3 difference, paired over 500 prompts | `+0.0645` | `[+0.030, +0.0995]` | **REVERSAL CONFIRMED** |

**My prediction was wrong, and it was wrong in the paper's favour.** I committed to
`REVERSAL UNRESOLVED` on the reasoning that the one order-averaging on record moved a gain by a
factor of `0.50`, so both arms would move together and the difference would fall under the
cross-pass floor. Both halves of that reasoning failed:

| arm | single order | order-averaged | ratio |
|---|---|---|---|
| selection, n=64 | `+0.110` | `+0.1045` | `0.95` |
| metered, k=10 | `+0.097` | `+0.0400` | `0.41` |
| difference | `+0.013` | `+0.0645` | `4.96` |

Order-averaging does not deflate the two arms together. It barely touches selection and removes
**59%** of the metered decoder's gain, so the difference does not shrink -- it grows five-fold. The
reviewer's extrapolation and mine were the same extrapolation, and the direction was the opposite
of what we both assumed.

A mechanism is available and we offer it as an explanation, not a finding: at `k=10` the metered
decoder is nearly the risky model it is being judged against (Section 5 measures it serving `p_r`
unchanged at every step it takes), so the judge is comparing two near-identical texts and falls
back on slot order. Selection's output is visibly a different text, so the judge has something to
read. Consistent with that, order consistency is lowest on the arms closest to their opponent.

**What this run does NOT do is replicate `+0.142` and `+0.072`, and it must never be quoted as if
it did.** It measures the same two arms under a *corrected and stricter* protocol, and the protocol
differs in three ways that are all improvements:

1. **One true prompt per item, for every arm.** `utility.py` and `selection_scaling.py` both show
   the judge a prompt reconstructed as `full_text` minus `generation`, which is not the prompt: the
   split point moves with how much each arm generated. For the same `prompt_id` the three arms here
   disagree on it in **455 of 500** cases. This run takes the prompt from the corpus, so all four
   arms are judged under identical conditions. See caution (aa).
2. **One seed per prompt** (the lowest), so the arms are paired item-by-item.
3. **Both orders**, which is the point of the arm.

The cost of correction (1) is visible and it cuts against the published number: under one true
prompt the **single-order** difference is `+0.013`, not the `+0.070` the published pair implies.
Most of the published single-order gap was the two arms being judged under different prompt
strings. The real difference is the order-averaged `+0.0645`, and it is real only because
order-averaging strips a position bias that was flattering the metered decoder.

**Consequence, applied.** The manuscript's headline comparison moves to the order-averaged pair and
says so: selection `+0.1045 [+0.082, +0.128]` against the metered decoder `+0.0400
[+0.014, +0.0655]`, difference `+0.0645 [+0.030, +0.0995]`, paired over 500 prompts with position
removed by construction. That is a weaker point estimate than `+0.142` and much better evidence for
it, and it is the number a reader should hold us to. `results/order_averaged_h2h.csv`,
`results/order_averaged_h2h_per_prompt.csv`.

Order consistency on this pass: `0.266` (sel n=64), `0.254` (sel n=1), `0.350` (metered k=10),
`0.244` (anchor k=0) -- all UNUSABLE, which is why the gains are reported order-averaged and the
levels are not reported at all.
