# feat-161 --- Does a larger scorer lift the JUDGED gain, or is the saturation reading right?

Committed **before the arm runs**. Nothing above `## Scoring log` is edited afterwards.

## Why

feat-158 and feat-160 found that every reward turn-over this paper reports on a **judge-free** task
is a `7`B-scorer effect: at `14`B and `72`B nothing turns over on TriviaQA or CoTaEval and both
Spearmans invert. The appendix now says, in as many words, that this **does not** reach the judged
workload, because the judged scorer ladder stops at `7.6`B (`0.5/1.5/3/7.6`B, feat-117) and its
reading is **saturation**: only the first of the three adjacent steps at `n=64` separates, and a
`1.5`B scorer already buys `87%` of the `7.6`B gain.

Those two pictures disagree, and the paper currently carries both with a disclaimer between them.
This arm measures the missing rung. It matters because the headline `+0.1045` is a judged number at
a `7`B scorer, and a deployer choosing a reward model wants to know whether spending more on it
buys anything on ordinary prompts or only on factual ones.

## What runs

`analysis/scorer_scale.py` with **two further** scorers, `Qwen2.5-14B-Instruct` and
`Qwen2.5-72B-Instruct`, added to the ladder, and **all six judged in ONE pass**. Two rungs rather
than one, decided before anything ran: this paper cannot compare judged levels across passes at
all, so a second pass later could never be set against this one, and the judge-free ladder that
motivates this arm runs to `72`B. That is not a convenience and the script's own docstring says why:
this paper forbids quoting judged levels across passes, and the comparison here is *between
scorers*, so they must share a control, an opponent, a prompt set and a judging session. Nothing is
generated --- the `64` cached candidates per prompt are feat-088's and are re-scored, exactly as the
four existing rungs were.

## Gates

**G0 --- the replication gate, scored first, unchanged from feat-117.** `sel7b_n64`, `sel05b_n64`
and `metered_k10` must land within `+/-0.04` of feat-116's `+0.1075`, `+0.0220` and `+0.0400` ---
the judge's own measured cross-pass floor, not a tolerance chosen here. **If any misses, the arm is
a failed replication and no band below is quoted.** feat-117's pass cleared this by `0.001` on all
three, so the gate is known to be satisfiable (caution (as): a gate nothing can pass gates nothing).

**G1 --- the reward cache must be complete**: `32{,}000` scores for the new scorer, the same count
as every existing rung.

Adding a fifth scorer enlarges the set of distinct served completions, which per caution (ap)
shifts presentation order for nearly every item. That is exactly why G0 exists with the judge's own
floor as its tolerance, and why every number below is a **gain over a shared control** rather than
a level.

## Bands

**B1 --- the terminal drop for the new scorer**, `D_14b = g(64) - g(16)`, paired per prompt, read
on the same rule as every other rung: `TURNS OVER` if the interval excludes zero below, `FLAT` if
it contains zero, `RISES` if it excludes zero above.

**B2 --- the reading this arm exists for**, the paired within-pass differences
`g_14b(64) - g_7b(64)` and `g_72b(64) - g_7b(64)`, read together --- the verdict below is
SATURATION HOLDS only if **neither** excludes zero above, with the `2.0`-half-width marginality rule:

| outcome | verdict | consequence, fixed now |
|---|---|---|
| interval excludes zero **above** | **THE JUDGED LADDER DOES NOT SATURATE** | Appendix~I's saturation paragraph is rescoped to the four scorers it was measured on, the paper states that a larger scorer buys judged utility too, and the disclaimer that feat-158's result does not reach the judged workload is **replaced by a measurement**. |
| contains zero | **SATURATION HOLDS** | The saturation reading stands and is now measured over five scorers spanning `0.494`--`14`B. The two pictures are reconciled: scorer size binds on factual tasks and not on judged ones, which is a statement about the *task*, not the mechanism. |
| excludes zero **below** | **A LARGER SCORER HURTS** | Reported as is, and the overoptimisation story returns at a new scale on the judged axis. |

**We predict SATURATION HOLDS.** Judged preference on open-ended prompts is a far easier target
than factual correctness, and a `1.5`B scorer already buys `87%` of the `7.6`B gain. **Registering
this means a LIFT counts against us in two ways at once** --- it would falsify the saturation
sentence the paper has carried since feat-117, and it would mean the headline `+0.1045` is not the
best this mechanism can do, which we would then have to say.

## Excluded in advance

* Comparing any level in this pass against a level from the feat-117 pass. G0's three reference
  arms are the only cross-pass numbers read, and they are read as a gate, not as a result.
* Adding a seventh scorer if the two new rungs disagree with the other four.
* Quoting `B2` if `G0` fails, for any reason, including a plausible one.
* Re-using this arm to revise any judge-free conclusion: different workload, different question.

## Compute

One `14`B and one `72`B reward pass over `32{,}000` cached candidates each, plus one six-scorer
judging session on `500` prompts. The `72`B reward is sharded over two cards (caution (q)). No generation. Comparable `14`B reward passes took under `3` h. Well under the
`24`-GPU-hour threshold.

## Scoring log
