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
1,200 item-judgements (600 prompts x 2 arms).

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
