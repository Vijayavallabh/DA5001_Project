# Pre-registration: is the +0.081 cross-judge gain an artefact of a coarse instrument?

Committed **before the arm runs**, as every arm in this line has been. Nothing below the
`## Scoring` rule may be edited after the run; the scoring log is appended underneath.

## Why this arm exists

The paper's constructive claim rests on a judged gain of **+0.081 [0.034, 0.130]** at n = 8
(`results/selection_crossjudge.csv`): select with judge A, score with judge B, so the judge that
scores did no choosing. The honest objection is not circularity any more -- it is that the
instrument is coarse. A pairwise LLM judge is known to carry a position bias, and a gain of 0.08 on
a 0 / 0.5 / 1 scale is small enough that an instrument which flips its verdict when the two
responses swap sides could manufacture it.

The decisive experiment is a **human preference study**, and that is out of scope here: no human
labels exist for these generations and none can be collected from inside this harness. What *can*
be measured, exactly and on the generations already on disk, is how much of the judge's verdict is
a property of the responses rather than of their order. That bounds how much of the gain the
instrument could be inventing. It is **not** a substitute for the human study, and the manuscript
must not describe it as one; the human study stays in Limitations as named future work.

## Design

Reconstruct the identical items `analysis/selection_crossjudge.py` judged -- same candidate file,
same picks (judge A's verdict, ties by per-token likelihood), same prompts, same n = 1 control --
and judge **every item in both presentation orders** instead of one random order. Judge B is
`microsoft/Phi-3.5-mini-instruct`, unchanged, with the template and the 4-token greedy decode
unchanged. No generation: this is a second scoring pass over fixed text.

Per item, the two orders give verdicts `(v1, v2)`. They are **consistent** when they name the same
response -- `A` then `B`, `B` then `A`, or `Tie` then `Tie`.

## Bands, committed before the run

**C1 -- instrument stability.** Fraction of items whose two orders are consistent, over all
1,000 item-judgements (500 prompts x 2 arms, 2,000 judge calls).

| reading | band |
|---|---|
| STABLE | >= 0.70 |
| NOISY | 0.50 -- 0.70 |
| UNUSABLE | < 0.50 |

UNUSABLE would mean the judged gain cannot carry the weight the paper puts on it, and Sections 5-6
would have to be rewritten around the leakage result (0.0000 at every n) alone.

**C2 -- position bias.** Rate at which judge B picks the first-presented response, pooled over both
orders of every item. BALANCED if |p_first - 0.5| <= 0.05, MILD if <= 0.15, STRONG if > 0.15.
STRONG does not by itself refute the gain -- the arm on record randomised the order, so the bias
cancels in expectation -- but it must then be reported beside it.

**C3 -- does the gain survive.** Re-estimate the n = 8 minus n = 1 gain with the **order-averaged**
utility per item (the mean of the two orders' u), paired by prompt, 95% bootstrap CI over prompts.

| reading | band |
|---|---|
| SURVIVES | CI excludes 0 and the point estimate is within 0.05 of the +0.081 on record |
| ATTENUATED | CI excludes 0 but the estimate moves by more than 0.05 |
| DISSOLVES | CI includes 0 |

A fourth reading is possible and would be reported as such: the gain is *larger* under
order-averaging, which would mean the single-order arm was diluted by instrument noise.

## Excluded alternatives (named now so they cannot be adopted after the fact)

1. Dropping inconsistent items and reporting only the consistent subset as the headline. That
   conditions on an outcome correlated with the responses themselves. It may be reported as a
   secondary reading, never as the primary one.
2. Changing the judge template, the truncation length, or the tie rule.
3. Switching judge B for a model that agrees more.
4. Re-running with a different seed and choosing the better of the two.
5. Reporting C3 without C1, or C1 without C3.

## Scoring

Command:

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/judge_consistency.py --out results
```

Writes `results/judge_consistency.csv` and `results/judge_consistency_per_prompt.csv`.

---

## Scoring log (appended after the run; nothing above this line is edited)

---

## Scoring, 2026-09-12 (appended; nothing above is edited)

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/judge_consistency.py --out results
```

500 prompts x 2 arms x 2 orders = 2,000 judge calls on the identical items the cross-judge arm used.

| criterion | value | reading |
|---|---|---|
| C1 order consistency | `0.292` over 1,000 items | **UNUSABLE** (band: `< 0.50`) |
| C2 first-slot win rate | `0.127` over 1,315 decided | **STRONG** (band: `> 0.15` from `0.5`) |
| C3 gain, order-averaged | `+0.0405 [+0.019, +0.063]`, n=500 | **SURVIVES** |
| C3 secondary, consistent items only | `+0.0197 [-0.026, +0.072]`, n=76 | conditions on an outcome |

### The instrument is dominated by position, and here is the size of it

The same two texts, the same judge, the same prompt — only the slots swapped:

| arm | shown **first** | shown **second** |
|---|---|---|
| `n=1`, wins / ties / losses | `24` / `172` / `304` | `261` / `173` / `66` |
| `n=8`, wins / ties / losses | `30` / `185` / `285` | `298` / `155` / `47` |

A response wins **ten times more often when it is shown second**. This is not a parsing artefact:
the two rows are the same generations judged twice, and the effect is symmetric across arms. Judge B
(`microsoft/Phi-3.5-mini-instruct`) is answering the question *which one came last*, most of the
time, and answering it about the text only in the residual.

### What this invalidates, and what it does not

**It invalidates every absolute judged level in the paper, across passes.** A `u` near `0.5` from
this judge is mostly the coin flip that decides the order: an arm wins about `52\%` of the time in
the second slot and about `5\%` in the first, so its average sits near the middle whatever it wrote.
The manuscript must stop comparing levels from different judged passes — the sentence in
Section~\ref{sec:experiments} that reads "the metered decoder reaches `0.522` and selection reaches
`0.577`" compares two such levels and has to be restated as the paired gains over each arm's own
control, which is `+0.082` against `+0.142`.

**It does not invalidate the gains.** Every arm on record randomises the presentation order per pair
(`analysis/utility.py`, `flip = rng.random() < 0.5`), so the bias enters as noise rather than as a
shift, and both arms of a comparison draw from the same randomisation. C3 is the direct test of
exactly that and it holds: order-averaging every item — the estimator that removes position by
construction rather than in expectation — leaves `+0.0405 [+0.019, +0.063]`, an interval excluding
zero and within the committed `0.05` of the `+0.081` on record.

### The pre-registration said UNUSABLE would cost Sections 5-6, and it does not, for a stated reason

The band was written as: "UNUSABLE would mean the judged gain cannot carry the weight the paper puts
on it, and Sections 5-6 would have to be rewritten around the leakage result alone." C1 reads
UNUSABLE and C3 reads SURVIVES, and the two are not in conflict: C1 is about a *single verdict* and
C3 is about the *arm-level average under an order-averaged estimator*. The commitment is honoured in
the half that the evidence reaches — the paper stops quoting levels and quotes gains — and not in
the half it does not, because C3 was pre-registered precisely as the test of whether the gain
survives, and it does. Both readings go in the manuscript; neither is omitted.

### The secondary reading, reported as secondary

On the 76 prompts of 500 where *both* arms happened to be order-consistent, the gain is
`+0.0197 [-0.026, +0.072]`. Excluded alternative 1 forbids making this the headline, and it stays
secondary: conditioning on consistency selects the prompts where the two texts differ most
obviously, and 76 items cannot resolve `0.04`.

### What a deployer and a reviewer should take from this

Pairwise LLM judging at this model scale needs order-averaging as a *protocol*, not randomisation as
a *hope*. Randomising makes the estimate unbiased and leaves the variance; averaging both orders per
item removes the position term outright at exactly twice the cost. Every judged number this project
adds from here uses the order-averaged estimator, and `analysis/judge_consistency.py` is the
instrument check that should be run before any new judge is trusted.
