# Pre-registration: is it the SIZE of G that breaks TokenSwap, or the MASS? (feat-169)

Committed **before any rung is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why

feat-167 found that TokenSwap's suppression is not a property of the method alone: at
`KL3M-170m` it leaks `0.2113` near-verbatim recall and `23`/`100` at ROUGE-L `>= 0.5`, where
`DistilGPT-2`, `Pleias-350m` and `TinyComma-1.8B` all read `0.0000` and `0`/`100`. The failing
rung keeps `171` of `431` token ids and `1.01%` of the main model's probability mass, and binds on
`1.79%` of steps against `25`--`29%`.

That reading is **correlational and confounded three ways**. `KL3M` is simultaneously the rung with
the fewest `G` ids, the least mass on `G`, the smallest vocabulary (`32{,}768`) and a different
training corpus, and `results/tokenswap_g_survey.csv` shows the cached tokenizers offer no way to
separate them: thirteen candidates give `|G| = 171` (all four `KL3M` sizes) or `397`--`431`
(everything else), with **nothing in between**. Vocabulary size and `|G|` move together across
every model we have.

So the ladder has to be built rather than found. **Hold the auxiliary fixed at `DistilGPT-2` ---
the one their own paper uses, and one that suppresses completely at full `G` --- and shrink `G`
itself.** That makes `|G|` a cause we set rather than a correlate we observe, and it is the only
way to learn whether a deployer whose tokenizer keeps two thirds of `G` is safe.

## What runs

`analysis/tokenswap_decode.py --arms tokenswap` with the new `--g-words`/`--g-seed` flags, which
subsample `G`'s **word** list before pairing --- never its token ids, because a word carries all
four of its surface forms and the rule acts on every one, so dropping ids directly would build a
`G` no tokenizer could produce. Everything else is feat-165's leakage half verbatim: the LoRA
memoriser against its own base, `--split attack_train --limit 100`, `--seed-tokens 20`,
`--max-new 200 --temperature 1.0`, no `--chat` (caution (at)). The shared rule-off control is
feat-165's and is not re-run.

Rungs, with `|G|` measured before any generation and fixed here:

| words | `|G|` token ids | seeds |
|---|---|---|
| `10` | `39` | `0`, `1` |
| `25` | `98` | `0`, `1` |
| **`44`** | **`174`** | `0`, `1` |
| `65` | `254` | `0`, `1` |
| `85` | `327` | `0`, `1` |
| `110` (full) | `426` | --- (control, one run) |

Two `G`-seeds at every shrunk rung, because a single random subset that happens to drop the
highest-mass words would be indistinguishable from a real effect.

**`44` words is the rung that matters**: `|G| = 174` against `KL3M`'s `171`. It is the same count
with a different auxiliary, the same vocabulary the method's own paper uses, and no corpus
difference.

## Gates, read before any band

- **G0 (the ladder is the ladder).** Each rung's printed `|G|` must equal the value tabled above
  exactly. These were measured from the tokenizers before this file was written; a disagreement
  means the subsample path is not building the `G` this registration describes.
- **G1 (the machinery does not change the full-`G` arm).** The `110`-word control, run through the
  same `--g-words` code path, must read `|G| = 426` and near-verbatim recall `0.0000`, reproducing
  feat-167's `DistilGPT-2` rung. If it does not, the subsample path has altered the rule and every
  rung below it is uninterpretable. **This is the gate feat-166 did not have** --- an arm whose
  instrument was never checked against a reading on record.

No gate is placed on the bind rate. feat-167's G1 required the rule to bind on more than `1%` of
steps and **certified** a rung binding at `1.79%` that leaked `23`/`100`; a threshold that low
gates nothing, and inventing a higher one now would be choosing it against data already seen
(caution (as)). The bind rate and the mass on `G` are **reported beside every rung** instead, so no
reading can be quoted without them.

## Bands, committed before the data

- **B1 --- where does suppression break?** Near-verbatim recall at each rung. **SUPPRESSES** at
  `<= 0.01` on both seeds, **LEAKS** above `0.05` on either seed, **PARTIAL** between. Reported as
  the largest `|G|` that leaks and the smallest that suppresses.
- **B2 --- is `|G|` sufficient to explain `KL3M`?** The `44`-word rung, `|G| = 174` against
  `KL3M`'s `171`. **If it LEAKS**, count is sufficient: the finding generalises past one tokenizer
  and the paper says a deployer needs `G` above some size. **If it SUPPRESSES**, count is *not*
  the cause, and whatever `KL3M` did wrong is not something `|G|` alone predicts --- the paper then
  has to name the other quantity, and the ladder's own mass column is the first place to look.
- **B3 --- the shape, with both covariates.** Recall against `|G|`, and recall against mass on `G`,
  each named (THRESHOLD / MONOTONE / NOISY) and each with the bind rate printed. If the two axes
  disagree about which rung is anomalous, that disagreement is the result.

## Excluded in advance

- Adding a rung after seeing where the break falls, or dropping the seed that disagrees with the
  other.
- Reading any band if G1 fails.
- Treating a leak at a shrunken `G` as a criticism of TokenSwap as its authors specify it. They
  run full `G` with `DistilGPT-2`, which suppresses completely and is the control here. What this
  arm can criticise is deploying the method with a tokenizer that silently delivers a fraction of
  `G`, which is feat-167's finding and is what a `|G|` threshold would make actionable.
- Comparing any judged number across passes; this arm judges nothing.

## What we predict

**B2: SUPPRESSES**, and therefore **count is not the axis --- mass is.** `KL3M`'s `171` ids carry
`1.01%` of the mass, but a uniform `40%` draw of `G`'s words should keep roughly `40%` of it,
around `0.15`, because mass is spread over the high-frequency words that dominate `G` and a random
subset keeps them in proportion. `KL3M`'s surviving surface forms are evidently the rare ones ---
the common variants failed to be single tokens there --- so at equal `|G|` it has `15x` less to act
on. If that is right, the `44`-word rung binds at roughly `0.11` and suppresses, and the honest
statement becomes *"TokenSwap needs enough probability mass in `G`, which `|G|` does not measure"*.

We predict the break, if the ladder reaches it, falls at the `10`-word rung (`|G| = 39`, expected
mass around `0.03`) and that `25` words is marginal. **We are predicting against the simpler
story**, which is that `KL3M` leaked because its `G` was small; we think that story is a
coincidence of count and will fail its own test at rung `44`.

## Compute

Host B, `11` runs of `100` passages, about `10` minutes each, across the cards `feat-168` is not
using. Two waves.

## Scoring log

### G0 defect, recorded 2026-09-22 while the rungs were still generating

**Written before any rung finished and before any recall number existed.** The seven wave-1 runs
printed their `|G|` during model load, which is when this was caught.

G0 says *"Each rung's printed `|G|` must equal the value tabled above exactly"* and the table gives
**one** `|G|` per word count while the same table specifies **two** `G`-seeds. Those cannot both
hold: a different random subset of words is a different set of surface forms and therefore a
different number of token ids. The tabled values were measured at seed `0`, and at seed `0` every
rung matches exactly:

| words | tabled | seed `0` | seed `1` |
|---|---|---|---|
| `110` | `426` | `426` | --- (control, one run) |
| `85` | `327` | `327` | `330` |
| `65` | `254` | `254` | `251` |
| `44` | `174` | `174` | `170` |

So **G0 PASSES on the column it was measured on and is unsatisfiable on the other**, by arithmetic
rather than by anything the runs did. The seed-`1` values sit within `1.2%` of their twins, which
is what a second uniform draw of the same size should give, so nothing about the ladder is wrong
--- the gate's wording is.

It is **not** repaired now. Widening a threshold after launch is how a gate stops being evidence
(caution (as)), and the honest record is that G0 was written to cover one draw and applied to two.
It is reported as PASS at seed `0`, NOT APPLICABLE at seed `1`, and the seed-`1` counts are printed
above so a reader can check them against the seed-`0` ones directly. The general rule, which
belongs with caution (v): **a gate that fixes a constant must be specified per arm, and an arm that
varies a seed varies every quantity derived from it.**

## Scoring, 2026-09-22

### Gates

| gate | requirement | measured | reading |
|---|---|---|---|
| G0 | each rung's `\|G\|` equals the tabled value | `426`/`327`/`254`/`174`/`98`/`39` at seed `0` | **PASS** (seed `0`; NOT APPLICABLE at seed `1`, see the defect note above) |
| G1 | full-`G` control reads `\|G\| = 426` and recall `0.0000` | `426`, `0.0000`, `0`/`100` at both thresholds | **PASS** |

G1 is the gate feat-166 did not have, and it does its job: the `110`-word run through the
`--g-words` code path reproduces feat-167's `DistilGPT-2` rung exactly, so the subsample machinery
has not altered the rule and every rung below it is interpretable.

### The ladder

One auxiliary throughout (`DistilGPT-2`, `82`M, theirs). `results/tokenswap_gsize.csv`.

| words | `\|G\|` | mass on `G` | binds | nv-recall | `lcs_word` | ROUGE-L `>= 0.5` | reading |
|---|---|---|---|---|---|---|---|
| `110` | `426` | `0.3746` | `29.62%` | `0.0000` | `2.95` | `0`/`100` | SUPPRESSES |
| `85` | `327` | `0.3287` | `25.07%` | `0.0000` | `3.38` | `0`/`100` | SUPPRESSES |
| `85` | `330` | `0.3130` | `25.14%` | `0.0000` | `3.10` | `0`/`100` | SUPPRESSES |
| `65` | `254` | `0.2807` | `21.68%` | `0.0000` | `3.62` | `0`/`100` | SUPPRESSES |
| `65` | `251` | `0.2797` | `21.68%` | `0.0000` | `3.59` | `0`/`100` | SUPPRESSES |
| **`44`** | **`174`** | `0.2021` | `14.21%` | **`0.0044`** | `5.51` | `0`/`100` | **SUPPRESSES** |
| **`44`** | **`170`** | `0.1850` | `13.68%` | **`0.0060`** | `5.73` | `0`/`100` | **SUPPRESSES** |
| `25` | `98` | `0.1258` | `7.79%` | `0.0394` | `11.15` | `3`/`100` | PARTIAL |
| `25` | `99` | `0.1199` | `7.37%` | `0.0339` | `10.77` | `1`/`100` | PARTIAL |
| `10` | `39` | `0.0396` | `2.32%` | `0.1905` | `34.33` | `21`/`100` | LEAKS |
| `10` | `40` | `0.0337` | `1.29%` | `0.2637` | `45.98` | `34`/`100` | LEAKS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `KL3M-170m` (feat-167, **held out**) | `171` | `0.0101` | `1.79%` | `0.2113` | `41.15` | `23`/`100` | LEAKS |

**B1.** The smallest `|G|` that suppresses on both seeds is `170` (mass `0.1850`); the largest that
leaks is `40` (mass `0.0396`). The transition is orderly and runs through a `PARTIAL` band at
`|G| ~ 98`, so this is a graded failure rather than a cliff: the two seeds agree on the reading at
every one of the five rungs, which is why two were run.

**B2 --- SUPPRESSES, and our prediction is CONFIRMED.** At `44` words the rule holds `|G| = 174`
and `170`, against `KL3M`'s `171`, and reads `0.0044` and `0.0060` --- both inside the `<= 0.01`
band --- where `KL3M` reads `0.2113`. Same count, same auxiliary, no corpus difference, **a factor
of `38` in the outcome.** So **count is not sufficient to explain `KL3M`**, the registered
consequence applies, and the paper must name the other quantity rather than saying `G` was too
small.

**B3 --- the shape.** Against `|G|` and against mass the ladder gives the **same** shape,
**MONOTONE in the rung means** and threshold-like: recall is identically `0.0000` above
`|G| = 250`, lifts off between `170` and `98`, and runs away below `40`. Monotone in the *means*
and not pointwise, which is stated because the exception is informative: the two seeds at the
`10`-word rung read `0.1905` and `0.2637`, a spread of `0.0732`, by far the widest on the ladder
(the next is `0.0055`). Where the rule barely fires, *which* words survive starts to matter as much
as how many --- and that is the same lesson the held-out point teaches, arriving from inside the
ladder. The two axes are monotonically related *along
the ladder*, by construction, so they cannot be told apart on it. B3's registered condition ---
*"If the two axes disagree about which rung is anomalous, that disagreement is the result"* --- is
what fires, and it fires at the held-out point rather than on the ladder.

### The held-out test, and it is **post hoc**

**Stated plainly: this check was not registered.** B3 registered shape-naming; predicting the
held-out rung from each axis is an analysis added after the ladder was read, and it is reported as
such. The `0.05` tolerance below is likewise post-hoc. What makes it worth anything is that
`KL3M`'s reading was committed in feat-167 **before this ladder existed**, on a different
tokenizer, so the point being predicted could not be tuned.

Prediction by linear interpolation on the ladder --- the dumbest estimator that respects the data,
no fit and no parameters --- clamping to the nearest endpoint outside its range:

| axis | `KL3M`'s value | predicted recall | actual | `\|err\|` |
|---|---|---|---|---|
| `\|G\|` | `171`, **inside** the ladder | `0.0056` (interpolated) | `0.2113` | **`0.2057`** |
| mass on `G` | `0.0101`, **below** the ladder | `0.2637` (clamped) | `0.2113` | `0.0524` |
| bind rate | `1.79%`, **inside** the ladder | `0.2287` (interpolated) | `0.2113` | **`0.0174`** |

**`|G|` is refuted, and that is the strong result.** `KL3M`'s `171` sits squarely inside the
ladder's range, so the axis gets a clean interpolation rather than an extrapolation, and it is
wrong by `0.2057` --- it predicts complete suppression where the truth is a quarter of the passages
recalled near-verbatim. No clamping, no edge effect, no excuse.

**The bind rate predicts it, to `0.0174`.** That is the best of the three and it is the quantity
the rule's mechanism actually acts through: `G` is an opportunity, the bind rate is the rule firing.

**Mass gets the verdict right and the magnitude wrong**, and we say so rather than rounding it into
a win: `0.0524` is outside the `0.05` we used, and its prediction is a *clamp*, because `KL3M`'s
mass sits below the lowest rung this ladder reaches (`0.0337`). **The ladder cannot test the mass
axis at `KL3M`'s mass.** Extending it would mean adding rungs after seeing where the break falls,
which this registration excludes in advance, so it is not done here.

**Our own prediction was half right and we record which half.** We predicted `B2` SUPPRESSES ---
correct, and it is the load-bearing call --- and we predicted the axis is **mass**. The data name
the **bind rate** instead, with mass a good proxy that overshoots at the one point we can test it.
Naming the right family and the wrong member of it is still being wrong about the member.

### What the paper should say

Not *"TokenSwap needs a large enough `G`"* --- that is the claim this arm refutes, at a factor of
`38`. The defensible statement is: **TokenSwap's suppression tracks how often the swap rule
actually fires, and an auxiliary's tokenizer can drive that rate to near zero while leaving `|G|`
looking healthy.** `KL3M` keeps `40%` of the token ids and `2.7%` of the mass, because the surface
forms its vocabulary happens to keep are the rare ones. A deployer who checks `|G|` will not see
this; a deployer who measures the bind rate will.
