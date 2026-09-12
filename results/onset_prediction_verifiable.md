# Pre-registration: is the constructive gain an artefact of the judge?

Committed **before the arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

Every utility number in this paper is a pairwise **judged** preference. The judge is dominated by
position (caution (m): the same two texts win 261 of 500 shown second and 24 shown first), which is
why Table 1 reports gains over each arm's own control rather than levels, and why every arm
randomises order per pair and is scored by two judges. Those repairs make the comparison fair. They
do not answer a different question a reviewer is entitled to ask: **is there any gain at all off the
judge's axis?**

GSM8K exact match is not a judge. The answer is a number and the grader is `==`. If selection
anchoring lifts it, the constructive claim is a property of the mechanism. If it does not, the claim
is a property of judged preference, and the paper must say so in Section 6 rather than let the
reader assume otherwise.

There is a second reason to run it, which is a contribution and not a control. **Majority vote ---
self-consistency, the most widely deployed inference-time method there is --- returns one of the `n`
samples drawn from the anchor, so `q(y) <= n p_s(y)` holds for the served string and it carries
exactly the same `\log n` pathwise certificate Proposition~4 proves.** Nothing in the proposition
mentions a reward. Measuring self-consistency here scores the certificate on a method practitioners
already run.

## The entry gate, read before the bands were written

A task the anchor cannot do measures nothing. The gate, fixed before it was read: the anchor's
greedy 8-shot accuracy on the first 100 test problems must lie in `[0.03, 0.90]`.

| anchor | greedy 8-shot, 100 problems | gate |
|---|---|---|
| `common-pile/comma-v0.1-2t` (7B) | `0.42` | **PASS** |
| `jacquelinehe/tinycomma-1.8b-llama3-tokenizer` | `0.04` | FAIL |

**The consequence is fixed here and is not favourable.** TinyComma is the only openly licensed
anchor that ships the Llama-3 tokenizer, so it is the only anchor at which the *metered* decoder can
run at all, and it cannot do this task. **There will therefore be no judge-free head-to-head against
the metered decoder**, and the paper may not imply one. This arm speaks to selection alone. The
head-to-head stays where it is: judged, at two pairs, in Section 6 and Appendix A.

## What is run

`common-pile/comma-v0.1-2t`, the anchor carrying the paper's largest judged gain (`+0.111
[+0.072, +0.148]`), on the **first 500 GSM8K test problems**, 8-shot with the eight training
problems as the fixed prompt, temperature `0.7`, `256`-token cap, `64` samples per problem, nested
by seed order so `n \in \{1,2,4,8,16,32,64\}` are prefixes of one draw and no sample is scored
twice. Two selection rules over the same samples:

1. **majority vote** on the extracted answer, ties to the earliest sample;
2. **the pointwise reward** the judged experiments use unchanged --- `log p("Yes") - log p("No")`
   from `Qwen2.5-7B-Instruct` on the one fixed template, which never sees the answer key and is not
   told the task is arithmetic.

Baselines are mandatory and on the same problems and seeds: `k = 0` is the anchor alone, which is
the `n = 1` arm, and `k = -1` is `Meta-Llama-3.1-8B-Instruct` alone, both greedy and sampled.

Accuracy is exact match on the last number of the completion, cut at any run-on `Question:`,
preferring the number after `####`. Intervals are a percentile bootstrap over problems, `10,000`
resamples; gains are **paired** against the same rule's own `n=1` on the same resampled problems.

## The metric gate

No band below is read if **more than 10%** of the `500 \times 64` samples yield no extractable
number. Above that the arm is measuring formatting, not arithmetic, and it is reported as
uninformative.

## Bands, committed before the run

**V1 -- does self-consistency lift it?** Majority-vote accuracy at `n=64` against its own `n=1`.

| reading | band |
|---|---|
| SC LIFTS | the paired 95% interval on the gain excludes zero |
| SC FLAT | it contains zero |

**V2 -- does the paper's own selector lift it?** The same, for the pointwise reward.

| reading | band |
|---|---|
| REWARD LIFTS | the paired 95% interval on the gain excludes zero |
| REWARD FLAT | it contains zero |

**V3 -- monotone in `\log n`?** Spearman of accuracy against `\log n` over the seven arms, per
rule. MONOTONE if `>= +0.8`, as it is on every judged arm (`+1.000` at this anchor).

**V4 -- the reading the paper must carry.** Fixed now, in all four cases, so none can be
renegotiated after the fact:

- **V2 fires.** Section 6 gains one sentence: the gain is not an artefact of judged preference,
  because the identical mechanism and the identical selector lift an exact-match metric with no
  judge in the loop.
- **V2 fails and V1 fires.** The certificate is fine and **the reward model is the weak link**: the
  paper must say the gain transfers to an objective metric only under a selector suited to the task,
  and name the reward model as the limit. This is a real weakening of the constructive claim's
  generality and it goes in Limitations, not in a footnote.
- **Both fail.** Section 6 must say the constructive gain is measured **only** under a judge and
  does not appear on an objective reasoning metric at the anchor where it is largest. The
  dichotomy --- the paper's actual claim --- is untouched either way: a metered per-token budget is
  vacuous or trivial whether or not any alternative works.
- **V1 fires and V2 fails, or both fire.** Either way the paper may state that self-consistency is a
  `\log n`-certified NAF mechanism, since that follows from Proposition~4 and not from this run.

**V5 -- descriptive, no band.** Where the served accuracy sits against the unconstrained
`Llama-3.1-8B-Instruct`. Reported for context; nothing is claimed from it, because the risky model
is instruction-tuned and the anchor is not, which is a confound this arm cannot break (caution: the
same base-vs-instruct confound that qualifies J1 in
`results/onset_prediction_imitation_70b.md`).

## Excluded alternatives

- Re-reading the bands on a different `n` grid, a different shot count, a different temperature or
  a different problem slice. The grid, the eight shots, `0.7` and the first `500` are fixed here.
- Swapping the reward model for one that is told the task is arithmetic, or for a verifier. That
  would be a different mechanism and would need its own pre-registration; it may not be used to
  rescue a FLAT reading.
- Reporting majority vote as though it were the paper's selector. It is a second instance of the
  same certificate and is reported beside the reward rule, never in place of it.
- Reading V1 or V2 on the risky model's own accuracy. `k = -1` is a baseline, not an arm.

## Scoring log
