# Pre-registration: does the judge-free climb survive a change of anchor?

**feat-137.** Written and committed **before the arm runs**. Nothing above the `## Scoring log`
heading is edited afterwards.

## Why

Every judged number in this paper is exposed to a judge that caution (m) measured as
position-dominated and caution (aa) found sharing a checkpoint with the opponent. The **judge-free
axis exists to make that objection answerable**, and it rests on **one anchor**. On Comma-7B, GSM8K
exact match over $500$ problems reads:

| rule | $n=1$ | $n=8$ | $n=64$ | gain at $64$ | Spearman |
|---|---|---|---|---|---|
| majority vote (self-consistency) | $0.320$ | $0.466$ | $0.542$ | $+0.222$ $[+0.182, +0.264]$ | $0.955$ |
| pointwise reward (Qwen2.5-7B) | $0.320$ | $0.364$ | $0.386$ | $+0.066$ $[+0.024, +0.108]$ | $0.929$ |

One anchor is not an axis. This arm adds the **second**, and it is chosen so that it answers a
question the paper already has open rather than merely adding a row: `common-pile/comma-v0.1-1t` is
the **same architecture at the same $7$B scale as Comma-7B, differing only in how much of the corpus
it saw**. `feat-136` is running that comparison on the *judged* axis right now (`comma1thb`). Running
it here too puts the **training-data ablation on both axes at once**, and the two can disagree.

## What is measured

`analysis/selection_verifiable.py` with the flags `scripts/run_verifiable.sh` used for the arm on
record, changing only the anchor and the tag: `--limit 500 --max-n 64 --batch-size 32
--reward-batch-size 16 --task gsm8k --n-shot 8 --max-new 256 --temperature 0.7 --seed 8801`.
Both selectors are scored, the same $500$ GSM8K problems, the same $8$-shot prompt, the same reward
model. **No judge is involved anywhere**: the metric is exact match against the reference answer.

| anchor | model id | tag |
|---|---|---|
| Comma-7B (1T tokens) | `common-pile/comma-v0.1-1t` | `_comma1t` |

The anchor has already **PASSED** leakage vetting on record (`anchor_vetting.csv`, `comma1t`,
100-token raw prefix, $0.0$ leaking), so no G1 is repeated here.

## G0 --- the floor gate, which decides whether a flat curve means anything

**An anchor that cannot do the task at all produces a flat curve, and a flat curve from a floor is
not a saturation.** This is the failure mode that would let a null be misread as a finding, and it is
gated before the fact.

**G0. The $n=1$ majority-vote accuracy must be $\ge 0.05$.** Below that, the arm is reported
**UNINFORMATIVE** --- not SATURATED, not a refutation of anything --- and the committed band below is
**not computed**. Comma-7B reads $0.320$; a same-size model trained on half the corpus reading under
$0.05$ would mean the task is out of reach, not that selection fails on it.

**G1. Coverage.** The grid must cover $n \in \{1,2,4,8,16,32,64\}$ on all $500$ problems under both
rules, and the $n=1$ row must be the anchor's own first draw. BLOCKS the band.

## Committed band --- the gain at $n=64$ under MAJORITY VOTE

Majority vote is the headline **because it has no scorer to overoptimise against** (the appendix
already says so, and caution (ao) records why: on TriviaQA the pointwise reward's accuracy *falls*
with $n$). The band is $\text{acc}(64) - \text{acc}(1)$ with its bootstrap interval, computed within
this pass over the same $500$ problems.

| reading | verdict |
|---|---|
| $>0$, interval excludes $0$ | **CLIMBS.** The judge-free climb is not specific to the 2T model, and corpus size does not gate the mechanism on a verifiable task. The axis becomes two anchors. |
| interval contains $0$ | **NO CLIMB.** At fixed architecture and scale, the amount of pre-training data gates the mechanism on the verifiable axis. |
| $<0$, interval excludes $0$ | **TURNS OVER.** Majority vote turning over would be the first instance anywhere in this paper of a *scorer-free* rule degrading with $n$, and Appendix~I's scoped no-overoptimisation sentence is corrected to name it. |

## The cross-axis reading, committed now because it is the point of the arm

`feat-136`'s `comma1thb` measures the **judged** paired $g(64)-g(8)$ at this same anchor. All four
joint outcomes are given a reading in advance, so none can be chosen after the fact:

| judged (`comma1thb`) | judge-free (here) | what it means |
|---|---|---|
| CLIMBS | CLIMBS | Corpus size gates neither axis; the $7$B result is about architecture and scale. |
| SATURATED | NO CLIMB | **The strongest H3 reading**: corpus size gates the mechanism at fixed size, and it does so on a task with no judge in it, so the finding cannot be blamed on the judge. |
| SATURATED | CLIMBS | The judged null at this anchor is an artefact of the judge or the scorer, not a limit of the anchor --- and the paper must say so, because it weakens every judged null it reports. |
| CLIMBS | NO CLIMB | **The most damaging reading, and it is reported as such**: a judged gain at this anchor that no verifiable improvement backs. It would not refute the certificate, which is a bound and not a utility claim, but it would put a named limit on what the judged gains mean. |

## Committed secondary, reported whatever it reads

* The **pointwise-reward** arm on the same grid, and specifically whether majority vote beats it. On
  record the two gains are $+0.222$ against $+0.066$, a factor of $3.4$; if that ordering **inverts**
  here, the appendix's reason for making majority vote the headline is anchor-specific and must be
  scoped to the anchor it was measured on.
* Spearman of accuracy against $\log n$ under both rules, descriptive.
* The risky model's own $k=-1$ accuracy, greedy and sampled, on the same $500$ problems --- the
  ceiling the anchor is being compared against, and the reason no level here is quoted alone.

## Excluded in advance

* Reading the band if G0 or G1 fails. **When a gate fails the band is not computed "just to see"**
  --- `feat-132`'s never was, which is the only thing that let its re-run be honest.
* Quoting the pointwise-reward arm as the headline if majority vote disagrees with it.
* **Comparing accuracy LEVELS across the two anchors as though the difference were a finding.** They
  are different models; only each arm's own gain over its own $n=1$ is read, which is the same
  discipline Table 1 adopted when caution (m) made levels uninterpretable.
* Re-running at another seed and keeping the pass that agrees with Comma-7B.
* Reading TriviaQA here. It is a separate task with its own committed table and its own known
  turnover; adding it after seeing GSM8K would be choosing the task after the answer.

## What this arm cannot do, stated before it runs

It measures one further anchor on one verifiable task with one prompt format, and $500$ GSM8K
problems at $8$ shots is a narrow window on "reasoning". A NO CLIMB here is a statement about
Comma-7B-1T on GSM8K and not about verifiable tasks in general. It cannot separate a corpus-size
effect from any other difference between the two checkpoints that the model cards do not disclose.
And it says nothing about extraction or about the certificate, which is a bound that holds whatever
the accuracy does.

## Compute

$500$ problems $\times\ 64$ draws $= 32{,}000$ generations at $7.0$B and `--max-new 256`, against the
$13.42$ gpu-hours `compute_hours.csv` records for $32{,}000$ at $7.6$B and $200$ tokens. Scaling by
parameters and tokens gives $\approx 15.8$ A100-hours, and an H100 is taken at only $1.5\times$ an
A100 (conservative), so **$\approx 10.5$ gpu-hours of generation plus the reward pass and the risky
baseline, call it $\approx 13$ gpu-hours on one card.** **Under the $24$-gpu-hour escalation
threshold.** It runs on the second host, on a card freed by a `feat-136` arm, under the user's
standing instruction of 2026-09-19 to use all eight of its GPUs.

## Scoring log

### 2026-09-19 --- correction: the compute estimate above was extrapolated when a direct measurement existed

**The estimate is not edited.** It sits above the `## Scoring log` heading and this file promises on
line 3 that nothing there is touched afterwards, so the number the arm was registered with stands and
the correction is recorded here instead.

The paragraph scales the *breadth* arm's $13.42$ gpu-hours by parameters and tokens to
$\approx 15.8$ A100-hours, then to $\approx 13$ gpu-hours on one card. But
`results/compute_hours.csv:314` already holds `verifiable_comma7b` --- **the same script, the same
$500$ problems, the same `--max-n 64`, on a slightly LARGER anchor** --- at **$6.88$ gpu-hours**,
measured. The registered estimate is therefore about $2.3\times$ too high, and the right figure for
this arm is **$4$--$5$ H100-hours**.

The direction is conservative, so the registered conclusion --- **under the $24$-gpu-hour escalation
threshold** --- is not merely unaffected but now holds with a much wider margin; no permission
question turns on it. What the error cost was accuracy, not safety. The habit it breaks is caution
(v)'s: **read the measurement of the thing you are about to run** before extrapolating from a
neighbour, and prefer a row of `compute_hours.csv` that names the same script to any scaling argument.

Found during the 2026-09-19 adversarial pass and recorded there first
(`results/onset_prediction_breadth_ladders.md`); it belongs here, in the file whose own number was
wrong, which is why it is repeated rather than cross-referenced.
