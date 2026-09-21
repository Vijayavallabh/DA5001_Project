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
