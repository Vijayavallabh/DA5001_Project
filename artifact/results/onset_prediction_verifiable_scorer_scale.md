# Pre-registration: does scorer-scale saturation survive without a judge?

Committed **before the arm runs**, as every arm in this line has been. Nothing above the
`## Scoring log` rule may be edited after the run; the scoring log is appended underneath.

## Why this arm exists

feat-117's most useful result is that **scorer capability saturates early**: at `n=64` a `1.5`B
reward beats a `0.5`B one by `+0.0700\,[+0.0500,+0.0900]` while `3`B over `1.5`B and `7.6`B over
`3`B buy `+0.0040` and `+0.0095` and neither separates. From that the paper now says the
`61.3\times` serving cost is the price of the scorer we chose and not of the mechanism.

Every number in that sentence comes from a pairwise LLM judge whose order consistency this paper
measures at `0.24`--`0.35` and calls UNUSABLE. The gains are taken over a shared control with
position removed by construction, and feat-117's G0 reproduced three reference arms to `0.001`
across independent passes, so they are stable --- but stable is not the same as valid. A judged
preference could saturate in scorer scale because *judging* saturates.

There is an axis with no judge on it and the candidates are already on disk. This runs the same
comparison on GSM8K exact match, where the answer is a number and the grader is `==`.

## Design

**No generation and no judge.** `output/phase5/verifiable/anchor_comma7b_n64.jsonl` holds 500 GSM8K
problems x 64 Comma-7B samples, `anchor_tqa_comma7b_n64.jsonl` the same for TriviaQA, and
`results/selection_verifiable_rewards_comma7b.csv` is the `7.6`B reward over all 32,000 GSM8K
candidates. Phase 1 re-scores those same cached candidates with `Qwen2.5-0.5B-Instruct`
(`0.4940`B), `Qwen2.5-1.5B-Instruct` (`1.5437`B) and `Qwen2.5-3B-Instruct` (`3.0859`B) --- same
family, same chat template, same `log p("Yes") - log p("No")` on the same fixed template, so scale
is again the only thing that varies. Phase 2 reads exact match at each `n` on the grid
`{1,2,4,8,16,32,64}`, bootstrap over problems, 10,000 resamples, paired wherever two arms are
compared.

`analysis/selection_verifiable.py` gained one argument, `--reward-tag`, which names the reward cache
and the output CSV without touching the generation paths, so a second scorer re-scores the same
generations instead of drawing new ones. The same patch fixed a latent defect: the reward arm's
label was the string `"pointwise reward (Qwen2.5-7B)"` hardcoded, so `--reward-model` would have
silently mislabelled every row it produced. It is now derived from the checkpoint. With the
defaults the patched script reproduces `results/selection_verifiable_comma7b.csv` **byte for byte**,
which was verified before this file was committed.

## What is and is not on trial

The certificate is not, as always: every arm serves one of `n` anchor draws, so `q(y) <= n p_s(y)`
and the budget is `\log n` whatever the scorer. Majority vote is carried alongside as the
**scorer-free** reference --- it is self-consistency, it needs no reward model at all, and on this
workload it already beats every reward arm --- but it is not a point on the scorer-scale axis and
is not treated as one.

**Shape is not the test here.** On this axis the `7.6`B reward is *already* non-monotone: its
Spearman against `log n` is `0.9286`, with dips at `n=4` and `n=16`, and even majority vote falls
from `0.546` at `n=32` to `0.542` at `n=64`. A wobbly small-scorer curve would therefore show
nothing. The test is **levels at `n=64`**.

## Bands, committed before the run

* **H1 -- does scorer scale matter at all without a judge?** `acc_{7.6B}(64) - acc_{0.5B}(64)`,
  paired over the 500 problems. `SEPARATES` if the 95% CI excludes 0, `FLAT` if it contains 0.
  Committed expectation: **SEPARATES**, in `[0.02, 0.08]`.
* **H2 -- the three adjacent steps at `n=64`**, each paired: `1.5`B over `0.5`B, `3`B over `1.5`B,
  `7.6`B over `3`B. Each reads `SEPARATES` or not.
  Committed expectation, mirroring feat-117 exactly: **the first separates and the two above it do
  not.**
* **H3 -- the headline. Do the two axes agree?**
  * `AGREES` --- the set of adjacent steps that separate is exactly \{`1.5`B over `0.5`B\}.
  * `DISAGREES, LATER SATURATION` --- some step above `1.5`B separates.
  * `DISAGREES, NO SCORER EFFECT` --- no adjacent step separates and H1 reads `FLAT`.
  * `DISAGREES, OTHER` --- any other pattern.
  Committed expectation: **`AGREES`**.
* **H4 -- TriviaQA, committed here and not added later.** The same four scorers and the same
  estimands on `anchor_tqa_comma7b_n64.jsonl`. The paper already reports the `7.6`B reward
  *losing* accuracy with `n` on this task (`-0.038\,[-0.068,-0.008]` at `n=16`), so the scorer-scale
  contrast may be absent or inverted. It is named now precisely so that it cannot become a
  post-hoc rescue: whatever it says is reported beside GSM8K and **does not change the H3 reading**.
* **H5 -- shape, descriptive.** Spearman(accuracy, `log n`) per scorer per task, reported against
  the `7.6`B scorer's own `0.9286` on GSM8K as the reference for what monotone means on this axis.
  No threshold is committed and none will be read.

## What each reading costs, committed in advance

* **`AGREES`** --- the saturation result is stated on both axes and Limitations cites the judge-free
  one, because it does not depend on the instrument Section 3 distrusts. This is the outcome that
  would make it the strongest claim in the paper.
* **`DISAGREES, LATER SATURATION`** --- the judged saturation is instrument- or workload-specific.
  The appendix says the two axes disagree, gives both, and the `87\%`/`35\%` sentence is qualified
  to the judged workload in Section 2 and Limitations.
* **`DISAGREES, NO SCORER EFFECT`** --- the judge-free axis does not reproduce the scorer-scale
  effect at all, which is evidence against the judged result. **The `87\%`/`35\%` claim comes out of
  Section 2 and Limitations**, the paper reverts to conceding `61.3\times` without the saturation
  escape, and the appendix reports both axes and says which one failed. This is the expensive
  reading and it is why the arm is worth running.
* `DISAGREES, OTHER` --- reported as measured, with the pattern named, and the `87\%`/`35\%`
  sentence qualified as under `LATER SATURATION`.

## Alternatives excluded in advance

* No fifth scorer and no second family. If the boundary falls between two of these four, the paper
  says it falls between them.
* No other `n` grid, no other anchor, no re-generation: the candidate pools are the two files named
  above exactly as they stand.
* TriviaQA will not be promoted over GSM8K if GSM8K disagrees with the judged result, and GSM8K
  will not be promoted over TriviaQA if TriviaQA is the one that agrees. GSM8K is primary because
  its grader is exact equality on a number; that is fixed here, before either is read.
* Majority vote will not be reported as a scorer scale, and its own non-monotonicity will not be
  used to excuse a reward arm's.
* The single-order or judged statistics will not be reintroduced to reconcile a disagreement.

## Scoring log

## Scoring, 2026-09-15

**Run.** `output/logs/verifiable_scorer_scale.log`, GPU 4, 2026-09-15, queued by
`scripts/run_verifiable_scorer_scale.sh`. Six reward passes over the two cached candidate pools;
no generation and no judge. Bands computed by `analysis/verifiable_scorer_scale.py`.

**H0 `MATCHES`.** The accuracy recomputed here from the caches equals what
`selection_verifiable.py` committed, for every scorer, task and `n`. A second, unplanned check came
free: **majority vote never touches a reward model, so all four per-scorer runs must reproduce it
exactly, and they do** --- `0.320/0.320/0.392/0.466/0.500/0.546/0.542` on GSM8K in every one. That
pins the whole cached-generation path, not just the arithmetic.

### GSM8K, the primary task

| scorer | `n=1` | `2` | `4` | `8` | `16` | `32` | `64` | Spearman |
|---|---|---|---|---|---|---|---|---|
| `0.5`B  | `0.320` | `0.336` | `0.320` | `0.302` | `0.312` | `0.332` | `0.316` | `-0.3243` |
| `1.5`B  | `0.320` | `0.330` | `0.316` | `0.318` | `0.332` | `0.324` | `0.310` | `-0.2143` |
| `3`B    | `0.320` | `0.334` | `0.334` | `0.320` | `0.338` | `0.344` | `0.356` | `+0.8365` |
| `7.6`B  | `0.320` | `0.352` | `0.344` | `0.364` | `0.356` | `0.380` | `0.386` | `+0.9286` |
| majority vote | `0.320` | `0.320` | `0.392` | `0.466` | `0.500` | `0.546` | `0.542` | `+0.9550` |

| band | value | 95% CI | reading |
|---|---|---|---|
| H1 `7.6`B `-` `0.5`B at `n=64` | `+0.0700` | `[+0.0300, +0.1120]` | **SEPARATES** |
| H2 `1.5`B over `0.5`B | `-0.0060` | `[-0.0500, +0.0360]` | does not separate |
| H2 `3`B over `1.5`B | `+0.0460` | `[+0.0000, +0.0920]` | does not separate |
| H2 `7.6`B over `3`B | `+0.0300` | `[-0.0140, +0.0740]` | does not separate |
| H3 | -- | -- | **DISAGREES, OTHER** |

### TriviaQA, the secondary committed in advance

H1 is `+0.0040 [-0.0300, +0.0380]`, **FLAT**; no adjacent step separates; H3 reads
**DISAGREES, NO SCORER EFFECT**. Every reward arm *declines* in `n` --- Spearman `-0.3063`,
`-0.9910`, `-0.7748`, `-0.6071` at `0.5`/`1.5`/`3`/`7.6`B --- while majority vote rises to `0.334`
at `+0.9820`. The paper already reported the `7.6`B reward losing accuracy here; at no scale does a
reward model help on this task. Per the band as written, this is reported beside GSM8K and **does
not change the H3 reading**, which is GSM8K's.

## What this means, stated plainly

**My committed expectation was `AGREES` on both bands. Neither task agrees.** The saturation
pattern feat-117 measured with a judge does not reproduce on either judge-free task.

The two axes do not, however, disagree about *everything*, and the part they agree on is worth as
much as the part they do not:

| | judged (feat-117) | judge-free (GSM8K) |
|---|---|---|
| total effect of scorer scale, `7.6`B over `0.5`B at `n=64` | `+0.0835` | `+0.0700 [+0.0300,+0.1120]` |
| bought in the first step, `1.5`B over `0.5`B | `+0.0700 [+0.0500,+0.0900]` | `-0.0060 [-0.0500,+0.0360]` |
| any single adjacent step resolves | yes, the first | **no, none of the three** |
| capability still rising at `7.6`B | no | **yes** |

**Both axes agree that scorer scale is worth roughly seven to eight points end to end, and they
disagree completely about where in the range it is bought.** On the judged workload the whole
effect sits in the step from `0.5`B to `1.5`B and nothing above it separates --- saturation. On
GSM8K the first step buys nothing, no single step resolves at 500 problems, and the total accrues
gradually with the largest scorer still the best. `3`B over `1.5`B is the closest to resolving and
its interval stops exactly at zero, `[+0.0000, +0.0920]`, which is a boundary and not a finding.

The practical consequence matters more than the statistics. feat-117's result was about to become a
deployer-facing recommendation --- *use a `1.5`B scorer, keep `87\%` of the gain, pay `35\%` of the
cost*. A deployer whose workload looks like GSM8K would be badly served by that advice: at `1.5`B
they would get nothing at all. **The saturation claim is workload-dependent, we do not know which
behaviour a new workload will show, and the paper now says so where the recommendation is made.**

## Consequence, applied

Per the committed table, GSM8K's `DISAGREES, OTHER` takes the `LATER SATURATION` action: the pattern
is named, both axes are reported, and the `87\%`/`35\%` sentence is **qualified to the judged
workload** in Section 2 and Limitations rather than stated as a general property of the mechanism.
It is not withdrawn --- that consequence was reserved for `NO SCORER EFFECT` on the primary task,
and GSM8K's H1 separates, so scorer scale does matter without a judge.

Two things are strengthened rather than weakened by this. The first is the paper's central claim,
which this arm confirms without an instrument: **utility is the scorer's and the certificate is the
mechanism's** --- `+0.0700 [+0.0300,+0.1120]` of GSM8K exact match turns on the scorer alone, with
`\log n` identical throughout. The second is majority vote, which needs no reward model, reaches
`0.546` where the best reward reaches `0.386`, and carries the same `\log n`: on a task with a
checkable answer the scorer-free rule is the better one, and this arm makes that comparison at four
scorer scales instead of one.

## What does not change

The certificate, as always: every arm serves one of `n` anchor draws, so `q(y) <= n p_s(y)` holds
and the budget is `\log n` whatever the scorer costs or achieves --- `4.159` nats at `n=64` across
all 56 cells above.
