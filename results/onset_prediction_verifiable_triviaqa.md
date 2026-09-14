# Pre-registration: does the judge-free gain hold on a task that asks the anchor to KNOW?

Committed **before the arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

`results/onset_prediction_verifiable.md` showed the constructive gain is not an artefact of the
judge: on GSM8K, with exact match as the metric, majority vote took Comma-7B from `0.320` to `0.546`
and the paper's own pointwise reward gained `+0.066` `[+0.024, +0.108]`. That is **one task**, and it
is a reasoning task.

Proposition 4 bounds the served string to one the anchor would have drawn, so the mechanism can
re-rank what the anchor produces and cannot manufacture what it does not have. On GSM8K an anchor
that can do the arithmetic *sometimes* has correct draws to promote. A closed-book knowledge task is
the other case: if the anchor does not know who wrote a book, no `n` will make it know.
**TriviaQA is therefore the test that can separate "selection works" from "selection works on
reasoning".**

## The entry gate, read before the bands were written

The same gate as the GSM8K arm: greedy few-shot accuracy on the first 100 validation questions must
lie in `[0.03, 0.90]`.

| anchor | greedy 5-shot, 100 questions | gate |
|---|---|---|
| `common-pile/comma-v0.1-2t` (7B) | `0.19` | **PASS** |
| `jacquelinehe/tinycomma-1.8b-llama3-tokenizer` | `0.07` | **PASS** |

**TinyComma passes here and failed on GSM8K (`0.04`).** That is worth recording because it reopens
something the GSM8K arm had to rule out: TinyComma is the only openly licensed anchor with the risky
model's tokenizer, so it is the only anchor the *metered* decoder runs at, and a judge-free
head-to-head is therefore possible on this task and was not on GSM8K. **This arm does not run it.**
It is a separate arm needing its own pre-registration and its own corpus build, and folding it in
here would be scope creep past the bands below.

## What is run

`common-pile/comma-v0.1-2t` --- the same anchor as the GSM8K arm, so the two are directly
comparable --- on the **first 500 TriviaQA `rc.nocontext` validation questions**, 5-shot with the
first five training questions as the fixed prompt, temperature `0.7`, `24`-token cap, `64` samples
per question, nested so `n \in \{1,2,4,8,16,32,64\}` are prefixes of one draw. The same two
selection rules over the same samples: **majority vote** (ties to the earliest sample) and the
**pointwise reward** the judged experiments use unchanged.

Baselines are mandatory and on the same questions and seeds: `k = 0` is the anchor alone (the `n=1`
arm) and `k = -1` is `Meta-Llama-3.1-8B-Instruct` alone, greedy and sampled.

**Scoring is containment, applied identically to every arm**: a gold alias appears as a whole-word
span in the normalised answer line. Exact match would score by *format* rather than by knowledge ---
a base model completing a few-shot prompt emits `David Seville` and an instruction-tuned one emits
`The answer is David Seville` --- and a smoke run scored the latter `0/8` on questions it had right.
The `24`-token cap bounds what a verbose completion can sweep up by accident. This choice is fixed
here, before any number, and applies to the anchor draws, the selected outputs and both baselines.

## The metric gate

No band is read if more than `10%` of the `500 \times 64` samples yield no answer line at all.

## Bands, committed before the run

**W1 -- does self-consistency lift a knowledge task?** Majority vote at `n=64` against its own `n=1`.

| reading | band |
|---|---|
| SC LIFTS | the paired 95% interval on the gain excludes zero |
| SC FLAT | it contains zero |

**W2 -- does the paper's own selector?** The same, for the pointwise reward.

**W3 -- monotone in `\log n`?** Spearman of accuracy against `\log n` over the seven arms, per rule.
MONOTONE if `>= +0.8`, as it was on GSM8K (`+0.955` and `+0.929`).

**W4 -- the prediction, committed so the arm can refute it.** We expect **SC LIFTS, by less than on
GSM8K** --- selection re-ranks draws, and a knowledge question offers less spread between draws than
a multi-step derivation, where one arithmetic slip decides the answer. Concretely: the majority-vote
gain at `n=64` is predicted **below** GSM8K's `+0.222`. If it is *larger*, the prediction is wrong
and Section 6 says so rather than quietly reporting the bigger number.

**W5 -- the reading the paper must carry**, fixed now in all cases:

- **Both lift.** Section 6's judge-free sentence covers two tasks of different kinds, reasoning and
  knowledge, and says so in those words.
- **W1 lifts, W2 flat.** The reward model is the limit again, as on GSM8K where the two rules
  differed by `3.4\times`; Limitations keeps "the scorer binds before the anchor does" and gains a
  second task's worth of evidence.
- **Both flat.** The judge-free claim is **scoped to reasoning** in Section 6 and in the abstract:
  selection re-ranks what the anchor already produces and does not supply knowledge it lacks. This
  is the support ceiling in its sharpest form and it goes in the main text, not a footnote.

**W6 -- descriptive, no band.** The unconstrained `Llama-3.1-8B-Instruct`'s own accuracy. Reported
for context only; it is instruction-tuned and the anchor is not.

## Excluded alternatives

- Switching to exact match, or to a different containment rule, after seeing a number.
- Changing the shot count, the temperature, the cap, the `n` grid or the question slice.
- Running the judge-free head-to-head against the metered decoder inside this arm.
- Reporting only the rule that lifts. Both are scored and both are reported.
- Reading W1 or W2 on the `k=-1` baseline. It is a baseline, not an arm.

## Scoring log

## Scoring, 2026-09-14

Run: `scripts/run_verifiable_tqa.sh`, GPU 4, started 06:03, `[tqa] exit=0 at 07:39`. Outputs
`results/selection_verifiable_tqa_comma7b.csv` and `results/selection_verifiable_rewards_tqa_comma7b.csv`.

**Metric gate PASS.** `[verif] no answer extracted in 0.0133 of samples` --- 1.33% against the
registered ceiling of 10%.

| band | reading | evidence |
|---|---|---|
| W1 | **SC LIFTS** | majority vote at `n=64`: `+0.054 [+0.026, +0.082]`, excludes zero |
| W2 | **FLAT** | pointwise reward at `n=64`: `-0.014 [-0.046, +0.018]`, contains zero |
| W3 | **MONOTONE for SC, not for the reward** | Spearman `+0.982` against the registered `>= +0.8`; the reward is `-0.607` |
| W4 | **prediction HOLDS** | `+0.054` is below GSM8K's `+0.222`, as committed |
| W5 | **"W1 lifts, W2 flat"** fires | the consequence written before the run |
| W6 | descriptive | `Llama-3.1-8B-Instruct` alone: `0.722` greedy, `0.674` sampled |

**The reward is worse than flat.** W2's band is two-way and `n=64` lands in FLAT, but the arm is not
noise around zero: the reward's accuracy *falls* with `n` (Spearman `-0.607`), and at `n=16` the
interval excludes zero on the wrong side, `-0.038 [-0.068, -0.008]`. Selecting by the paper's own
scorer on a knowledge task is actively worse than taking the anchor's first draw. On GSM8K the two
rules differed by `3.4x` in the same direction; here they differ in **sign**.

**What this arm establishes and what it does not.** The judge-free claim now covers two tasks of
different kinds --- reasoning (GSM8K, `+0.222`) and closed-book knowledge (TriviaQA, `+0.054`) ---
so it is not an artefact of a judge and not an artefact of one task. Proposition 4's support
ceiling is visible in the size of the gain: a knowledge question offers less spread between draws
than a multi-step derivation, and the gain is `4.2x` smaller, which is the direction W4 committed
to. What the arm does **not** establish is that the mechanism's deployed scorer transfers. It does
not, on this task, at all.

**Consequences applied to the manuscript** (all three were fixed in W5 before the run):

1. Section 6's judge-free paragraph reports both tasks and both rules, including the negative one.
2. Limitations keeps *"the scorer binds before the anchor does"* and now cites two tasks: a `3.4x`
   gap on GSM8K and a **sign flip** on TriviaQA.
3. No abstract change. The abstract says "exact-match accuracy with no judge at all", which the
   majority-vote result supports on both tasks; it does not claim the pointwise reward transfers.
