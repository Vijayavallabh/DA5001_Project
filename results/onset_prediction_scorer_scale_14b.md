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

**Scored 2026-09-21.** One pass, six scorers, host B cards 0--1.
`scripts/run_scorer_scale_14b.sh`; CSVs `results/scorer_scale{,_bands,_per_prompt}.csv`.

### G0 REPLICATES, so everything below may be quoted

The three reference arms land inside the judge's own `+/-0.04` cross-pass floor. Adding two rungs
enlarged the set of distinct served completions to `5{,}263` across `36` selection arms, which per
caution (ap) shifts presentation order for nearly every item --- and the gains, taken over a shared
control with position removed by construction, replicate anyway. G1 (`32{,}000` scores per new
scorer) passes for both.

### The judged ladder at `n=64`, all six rungs, one pass

| scorer | gain at `n=64` | cost vs the meter |
|---|---|---|
| `0.494`B | `+0.0230` `[+0.0005, +0.0445]` | |
| `1.5437`B | `+0.0910` `[+0.0695, +0.1125]` | |
| `3.0859`B | `+0.1025` `[+0.0800, +0.1255]` | |
| **`7.6156`B** | **`+0.1095`** `[+0.0865, +0.1320]` | the paper's operating point |
| `14.7701`B | `+0.0810` `[+0.0600, +0.1030]` | |
| `72.7062`B | `+0.1010` `[+0.0780, +0.1235]` | `486.9x` |

**The ladder is NOT monotone in scorer size**, which neither we nor the registration anticipated.
It rises steeply to `1.5`B, is flat to `7.6`B, **drops** at `14.8`B, and recovers to statistical
parity at `72.7`B.

### B2, the registered readings, both against the `7`B rung

| difference | value | half-widths | registered row |
|---|---|---|---|
| `g_14b(64) - g_7b(64)` | `-0.0285` `[-0.0470, -0.0110]` | `1.58` | **A LARGER SCORER HURTS** (marginal) |
| `g_72b(64) - g_7b(64)` | `-0.0085` `[-0.0255, +0.0085]` | `0.50` | **SATURATION HOLDS** (marginal) |

The scorer emitted only ADJACENT steps (G4); both differences above are against the `7`B rung as
this file named them, computed with the same `paired_boot` and seed the scorer uses. For context
the adjacent step it did emit is `g_72b - g_14b = +0.0200` `[+0.0025, +0.0380]`, also marginal.

**We predicted SATURATION HOLDS, and in the sense that matters we were right: no rung above
`7.6`B lifts the judged gain.** The headline `+0.1045` stands as the best this mechanism does on
this workload, and a deployer buys nothing by spending more on the reward model --- at `n=64` a
`72.7`B scorer costs `486.9x` the metered decoder for a gain indistinguishable from a `7.6`B one's.
**All three readings are MARGINAL by this paper's own `2.0`-half-width rule** and are reported as
such; the dip at `14`B in particular is not a result we would build on.

### The two pictures are reconciled, which is what this arm was for

feat-158/160 found that on the **judge-free** tasks a larger scorer removes every turn-over and
inverts both Spearmans. Here, on the **judged** workload, a larger scorer buys nothing. Those are
not in conflict: scorer size binds where the task has a checkable answer and not where the target
is a judged preference on ordinary prompts, which a `1.5`B model already approximates well
(`+0.0910` of the `7.6`B rung's `+0.1095`). That is a statement about the **task**, not about the
mechanism, and it is exactly the consequence this file fixed in advance for SATURATION HOLDS.

### A defect in our own specification, recorded and not repaired (caution (w))

The `A LARGER SCORER HURTS` row's consequence says ``the overoptimisation story returns at a new
scale on the judged axis''. **G1 refutes that clause**: the `14`B scorer's own terminal drop is
`+0.0270` `[+0.0095, +0.0455]`, which **RISES** --- its curve does not turn over, it simply sits
lower. A lower level is not overoptimisation. The row's first half (``Reported as is'') is applied
and the overoptimisation clause is **withdrawn**. A future scorer-ladder arm should separate *level*
from *shape* in its consequence table, because this one conflated them.

### The rest of the grid, reported whatever it reads

G1 terminal drops: `0.494`B **FLAT** `-0.0165` `[-0.0345, +0.0010]`, `1.5437`B **FLAT**
`+0.0155` `[-0.0030, +0.0335]`, and RISES at `3.0859`B `+0.0345`, `7.6156`B `+0.0410`,
`14.7701`B `+0.0270`, `72.7062`B `+0.0310`. **G3: no turnover anywhere on the ladder.** G2 shape:
monotone at `1.5437`B, `3.0859`B, `7.6156`B and `72.7062`B; not monotone at `0.494`B (`0.5798`) or,
descriptively, at `14.7701`B (`0.9429`). G4 adjacent steps: only `1.5`B over `0.5`B
(`+0.0680`), `14`B over `7.6`B (`-0.0285`) and `72`B over `14`B (`+0.0200`) separate; the two
steps through the middle of the ladder do not.

### Two corrections made after the run, both recorded rather than quietly applied

**This pass overwrote a closed arm's CSV and was caught by three existing guards.** Running with
`--out results` wrote the canonical `scorer_scale{,_bands,_per_prompt}.csv`, which belonged to
feat-117's closed four-rung arm --- whose `+0.0700`, `+0.0040`, `+0.0095`, `87.3%`, `35.2%` and
`0.5429` the appendix still quotes. feat-117's files were restored from git, this pass now lives in
`scorer_scale_6rung*.csv`, and `scorer_scale.py` refuses to write over a different ladder unless
told to. Recorded as caution (ax). **No number in this log changed**: the six-rung pass is intact
and is what every figure above is read from.

**A parameter count was typed rather than counted.** `14.7701`B and `72.7062`B were entered from
knowledge into a list whose own comment says ``COUNTED OFF THE LOADED MODEL''. Counted off the
safetensors index (`total_size / 2`, a method validated by reproducing the committed `7.6156`
exactly for the `7`B rung), the `14`B scorer is **`14.7700`**B and the `72`B one `72.7062`B. The
constant and the two derived metadata cells were corrected by hand; **no measured quantity depends
on either value** --- they name a rung and scale the descriptive cost column, which carries no band.
