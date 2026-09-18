# Pre-registration: does Comma-7B's climb to $n=64$ survive an independent draw?

**feat-132.** Written and committed **before the arm runs**. Nothing above the `## Scoring log`
heading is edited afterwards.

## Why

`feat-131` has just narrowed the paper's breadth-at-$n=64$ claim to **two anchors**: TinyComma
(audited) and Comma-7B. It narrowed it because KL3M-1.7B's single climbing reading, `+0.0650
[+0.0270, +0.1030]`, moved to `+0.0040 [-0.0330, +0.0400]` under a disjoint seed draw --- a distance
of `0.061` where the audited anchor's own seed replication moved `0.0000`.

The lesson we drew from that pair, and wrote into caution `(ap)` and the appendix, is a **criterion**:
a paired difference is stable where the effect is large relative to its own interval and not where it
is marginal. It has exactly two data points, and both are on anchors whose fate was already known
when the criterion was written:

| anchor | paired $g(64)-g(8)$, judge B | half-width | $g$ / half-width | replicated? |
|---|---|---|---|---|
| TinyComma-1.8B (audited) | `+0.0880 [+0.0470, +0.1300]` | `0.0415` | **`2.12`** | yes, to four decimals |
| KL3M-1.7B | `+0.0650 [+0.0270, +0.1030]` | `0.0380` | **`1.71`** | no, moved `0.061` |
| **Comma-7B** | **`+0.1010 [+0.0590, +0.1420]`** | **`0.0415`** | **`2.43`** | **this arm** |

Comma-7B is the second of the two anchors the claim now rests on, it has never been re-drawn, and it
sits **above** the one ratio that replicated. So this arm does two things at once that no other
available arm does: it is the only outstanding check on a claim the paper now makes, and it is the
only **out-of-sample** test of the criterion the paper now states. The criterion predicts
**REPLICATES**. A pre-registration whose rule predicts the answer is worth running precisely
because the prediction can fail.

## What is measured

One change from the arm on record, the same change `feat-131` made:
`--seeds 42 43 44` $\rightarrow$ `52 53 54`. `build_trajectory_seeds` hashes the
`(prompt_id, seed, draw)` tuple into the high 16 bits, so the 64 trajectory seeds are **disjoint from
the original 64 by construction**, not by luck. Held fixed: the anchor
(`common-pile/comma-v0.1-2t`), the same 500 ordinary prompts, `--trajectories-per-prompt 64`,
`--max-new-tokens 200`, the reward, both judges, and the seven-point $n$ grid.

`--batch-size 32` is used, where the arm on record took `h1.py`'s default of `8`. **This is stated
now rather than discovered later.** Batch size is part of the seed (caution `(u)`), so it is a
second, independent re-roll of the *same* distribution on top of the seed change --- it cannot make
the two draws less disjoint, and it does not change what is being drawn from. It is the protocol
`feat-131` used, and holding it identical is what makes these two replications comparable, which is
the point of the arm.

Generation is split by prompt class across three cards and merged, exactly as `feat-131` and
`feat-129` were. That is validated empirically, not merely argued: `feat-129`'s class-split pool
reproduced the single-card pool's `32,000` rewards **bit-identically** at ranks `0`--`63`.

## There is deliberately NO bit-identity reproduction check

Same reason as `feat-131`, restated so this file stands alone: the arm is an independent draw *by
construction*, so nothing is supposed to match, and a bit-identity gate would be incoherent rather
than merely wrong. The integrity checks are distributional and are fixed now:

* the arm must cover the same `500` prompts, and its $n=1$ empty fraction must sit within `0.03` of
  the **`0.094`** the arm being replicated records --- measured on
  `output/phase5/sel_comma7b_64` with this scorer's own `empty_fraction`, not taken from a
  summary column;
* its $n=1$ judged **level** is not compared with any other pass (caution `(ap)`, and `G3` of
  `results/onset_prediction_selection_n64_comma7b.md`, which failed on exactly that comparison);
* and no number from this arm is set against a number from any other sweep except through the
  **paired** difference below, which is computed within one pass on both sides.

**That reference was wrong when first written, and the mutation test caught it before the run.**
The first draft of this file took `0.030` from `results/selection_breadth.csv`, which is the
**$n=8$** breadth arm's `empty_frac_n1`. That is a different draw --- this anchor's two arms differ
even in mean completion length, `99.5` words against `80.4`, and
`results/onset_prediction_selection_n64_comma7b.md` records `G3` failing on exactly that kind of
cross-arm comparison. The $n=64$ arm on record reads `0.094`, so the gate as first written would
have **failed a perfectly good replication** at `0.064` against a `0.03` tolerance. This is caution
`(v)`: a reference number carries its protocol, and a gate built on one without it is a gate built
on nothing. Recorded rather than quietly repaired, because `feat-131` wrote its reference the same
way --- from the $n=8$ column --- and passed only because that anchor's two arms happen to agree
(`0.002` against `0.000`). **The reference for a replication's integrity check must be measured on
the arm being replicated.**

## Committed bands --- read on the paired $g(64)-g(8)$, judge~B, 500 prompts

The same three-row table as `feat-131`, scored by the same code path, so that the two arms are read
by one rule and not by two.

| reading | verdict |
|---|---|
| $>0$ with its interval excluding $0$ | **REPLICATES.** Comma-7B's climb to $n=64$ survives an independent draw. The two-anchor breadth-at-$n=64$ claim is verified rather than merely surviving, and the stability criterion has an out-of-sample confirmation at `2.43` half-widths. |
| interval includes $0$ | **DOES NOT REPLICATE.** The climb at the second of the two retained anchors is a draw-level effect. The breadth-at-$n=64$ claim then rests on **the audited anchor alone**, and the appendix must say so in those words; the criterion is falsified out of sample and caution `(ap)` must record that a ratio above `2.4` did not protect a reading. |
| $<0$ with its interval excluding $0$ | **INVERTS.** Stronger than non-replication. The appendix must then say the $n=64$ reading at this anchor is not stable in the seed, and the paper makes **no** across-anchor claim at $n=64$ at all. |

## Committed secondary, reported whatever it reads

The distance $|D_{\text{rep}} - D_{\text{orig}}|$ between the two paired differences, against the two
precedents on record (`0.0000` at TinyComma, `0.0610` at KL3M-1.7B); the ratio $g$ / half-width in
the replication; the full seven-point grid under **both** judges; and the judged levels, which are
*expected* to move between passes and are reported as information rather than gated on.

## Excluded in advance

We will not, after seeing results: change the scorer, either judge, the opponent, the prompt set,
the batch size or the $n$-grid; quote judge~A if judge~B disagrees; compare any judged *level* across
passes; re-run at a third seed and pick the two that agree; pool the two draws; move the stability
criterion's threshold to whichever side of `2.43` the answer lands on; or treat a non-replication as
a reason to drop the anchor rather than to narrow the claim.

## What this arm cannot do, stated before it runs

It re-draws **one** anchor under **one** alternative seed triple. It says nothing about the two
Pleias anchors, nothing about KL3M-1.7B beyond what `feat-131` already read, and nothing about where
any ceiling sits --- Comma-7B's ceiling past $n=64$ remains unmeasured and is the arm we have not
run. It cannot separate a reward-model ceiling from an anchor-support ceiling. A replication that
succeeds raises the warrant on the second of two cells; it does not make two anchors a law, and the
paper will keep saying the climb past $n=8$ is established at two anchors and not in general.

## Compute

Measured from the arm on record, not estimated: `results/compute_hours.csv` has
`sel_comma7b_64` at **13.42 gpu-hours** for `32,000` trajectories on one card
(2026-09-12 20:27 $\rightarrow$ 2026-09-13 09:52, `0.662` traj/s at `--batch-size 8`) and its
scoring at **0.26 h**. Split by class across three cards, at that same measured rate:

| card | class | trajectories | expected |
|---|---|---|---|
| GPU 1 | neutral $200$ | $12{,}800$ | $\approx 5.4$ h |
| GPU 2 | creative $150$ | $9{,}600$ | $\approx 4.0$ h |
| GPU 4 | factual $150$ | $9{,}600$ | $\approx 4.0$ h |
| GPU 1 | merge + score | --- | $\approx 0.3$ h |

**Total $\approx 13.7$ gpu-hours**, under the `24`-gpu-hour escalation threshold, at $\approx 5.5$
hours of wall clock. `--batch-size 32` should beat the quoted rate; the estimate does not assume it.
Checked with `env -u LD_LIBRARY_PATH nvidia-smi` before launch: GPUs 1, 2 and 4 idle at `14` MiB;
**GPU 0 holds another user's `597` MiB and is not taken**; GPU 3 is the 4 GB T400 and is never used.

## Scoring log

## Scoring, 2026-09-18 (appended; nothing above is edited)

### **INVALID** --- the arm does not clear its committed integrity check, and the cause is a premise of ours

All three cards `rc=0`, merged and scored 16:33:06. The committed integrity check **fails**:

```
prompts: 500 (PASS)
n=1 empty fraction: 0.0560 against the committed 0.0940, tolerance 0.03 -> FAIL
Per the pre-registration, the arm does not clear its integrity checks. Chase it.
```

**The band was not read.** The scorer returns before computing it, by construction, and it has not
been computed by hand either --- so nothing below was chosen with knowledge of the answer, and the
corrected arm cannot be tuned to it. (What was visible in the merge log is the summary CSV's
judge-**C** levels and gains; the committed band is the paired $g(64)-g(8)$ under judge~B, a
different quantity, and it remains unread.)

### Chasing it, with the decision rule fixed before the band existed

The rule written down before any diagnostic was run: **concentrated in one prompt class $\rightarrow$
a pipeline bug, the arm is invalid and must be re-run; spread across all three $\rightarrow$ our
tolerance is mis-specified and the arm is read at a reduced warrant.** Per class, draw 0:

| class | arm on record (batch 8) | this arm (batch 32) | two-proportion $z$ |
|---|---|---|---|
| creative | $1/150 = 0.007$ | $0/150 = 0.000$ | $+1.00$ |
| factual | $1/150 = 0.007$ | $4/150 = 0.027$ | $-1.35$ |
| **neutral** | $\mathbf{45/200 = 0.225}$ | $\mathbf{24/200 = 0.120}$ | $\mathbf{+2.78}$ |
| total | $47/500 = 0.094$ | $28/500 = 0.056$ | $+2.28$ |

Empties are a **neutral-class phenomenon** at this anchor --- $22.5\%$ there against $0.7\%$ in the
other two --- and the entire gap is there, at $p \approx 0.005$. It is concentrated, so by the rule
above the arm is **invalid rather than reduced**.

### What it is concentrated *by*, which is the part worth keeping

Not a broken card: the $n=1$ arm's **mean completion length agrees to $0.4\%$** across the two passes
($99.2$ words against $99.6$ on the same $500$ prompts), which a wrong model, a wrong corpus or a
truncation would not survive. The two runs differ in exactly two things, and this file named both in
advance: the seeds, and `--batch-size 8` $\rightarrow$ `32`. About the second it asserted

> *"it is a second, independent re-roll of the same distribution --- it cannot make the two draws
> less disjoint, and **it does not change what is being drawn from**."*

**The second clause is false, and this arm's own gate is what falsified it.** An empty generation is
the model emitting end-of-text at step $0$; under left-padded batched generation the step-$0$ logits
depend on the padding pattern of the batch the prompt lands in, so batch size moves the empty rate
*systematically* where that rate is large and invisibly where it is near zero. That is why the effect
appears only in `neutral`, and it is why `feat-131` saw nothing: at KL3M-1.7B the quantity is
$0.000$--$0.002$ everywhere, so a $0.03$ tolerance there could not have detected a batch effect of
any size. Caution `(u)` says batch size is part of the seed; the stronger statement this arm supports
is that **at a rate-valued quantity it is not merely a re-roll but a shift**.

So the arm changed two things where the design intends one, and the second is now measured to move
the served distribution. It therefore cannot answer the question it was registered to ask --- not
because the replication failed, but because it is not the comparison it was meant to be.

### Consequence, and what this does NOT do

Per caution `(w)`, **a defect in our own specification must not retire a question by counting against
a stop rule.** This is not evidence for or against Comma-7B's climb; the question is exactly as open
as it was this morning. The breadth-at-$n=64$ claim stands where `feat-131` left it --- established
at TinyComma and Comma-7B, with Comma-7B's replication now *outstanding rather than failed* --- and
nothing in the manuscript changes on this arm's account.

The corrected arm is `feat-133`: same design, `--batch-size 8` so that **one** thing changes, bands
committed before it runs, and an integrity check whose tolerance is derived from the quantity's own
sampling variability rather than copied as a constant. It is registered in
`results/onset_prediction_comma7b_seed8.md`.

### The compute this cost, stated plainly

$\approx 6.0$ gpu-hours, spent on an arm that cannot be read. It was not wasted in one respect --- it
is what falsified the premise --- but the premise was ours to check before spending the cards, and
the check that would have caught it is the one this file's own gate performed afterwards.
