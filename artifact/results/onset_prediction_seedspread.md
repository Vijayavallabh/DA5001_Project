# Is the onset ratio reproducible under the paper's own recipe?

**Committed 2026-09-16 02:54, before any of these memorisers exists.** Nothing above the
scoring-log heading is edited after that line is written.

## The question feat-121 could not answer, and why it is worth a card

feat-121 held one pair, one corpus and one anchor fixed and varied `--epochs` alone. The onset ratio
moved from `0.8756` to `1.3477` --- a spread of `0.4721`, against the whole nine-pair table's
`0.2874`. It could not say whether the mover was **memoriser strength** or **run-to-run variation of
the fine-tune**, because four epoch counts are also four optimisation trajectories, and it forbade
itself from adding the control after the fact.

This is that control. `recipes/finetune_memorizing.py` has always taken `--seed`, and it seeds both
`random` --- which reshuffles the training texts every epoch (line 121) --- and `torch.manual_seed`,
which fixes the LoRA initialisation and dropout. **Every memoriser on record in this paper was
trained at `seed 0`**, including all nine pairs of the Section 4 table, so the paper has never
measured how much of an onset ratio is the seed.

## The design: the same corner, the orthogonal axis

Pleias-1.2B on BookMIA again, so the two ladders share a corner and are directly comparable. Three
new memorisers at **`--epochs 40`** --- the recipe every published memoriser in this paper uses, and
the one feat-120 ran --- differing from feat-120's only in `--seed` (`1`, `2`, `3`). Its `seed 0`
run is the fourth point, already measured and bootstrapped.

```
--target-modules all-linear --no-chat --epochs 40 --lr 3e-4 --rank 128
--batch 2 --accum 4 --max-len 0 --stop-loss 0.02      --seed 1 | 2 | 3
```

The epoch count is **not** chosen after seeing feat-121's four points. It is `40` because that is
what the paper's own recipe is; picking the epoch count that produced feat-121's extreme would be
choosing the answer.

Same corpus, same 100 swept passages, same seed for the sweep, and the same grid feat-121 used ---
BookMIA's committed grid with feat-120's licensed extension folded in, so all four points share one
grid and no crossing can sit at a ceiling:
`-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2 4.6 5.3 6.6`, `--modes single --limit 100`.

## Committed bands

The quantity is the **seed-only ratio span** over the four points, read against feat-121's
epoch-only span of `0.4721` on the identical pair, corpus, anchor and grid.

| outcome | reading |
|---|---|
| seed-only span **`>= 0.20`** | **it is run-to-run variation.** The onset ratio is not reproducible under this paper's own recipe, and the nine-pair table's `0.2874` of between-pair structure sits inside the noise of a single pair re-trained |
| seed-only span **`< 0.10`** | **it is strength.** The recipe reproduces at a fixed epoch count, so feat-121's `0.4721` belongs to what the epochs changed, and the nine-pair table is confounded by a variable it never controlled |
| anything else | **inconclusive**, reported as inconclusive. No second seed set, no fifth seed |

**Secondary, committed now so it is not post hoc.** Pooling the two ladders gives seven distinct
(memoriser, ratio) points on one pair and one corpus --- feat-121's four and this arm's three new
ones, sharing the `seed 0`, `40`-epoch corner. We report Spearman between sampled `k=-1` and the
onset ratio over those seven with its exact permutation `p` (floor `1/2520` at `n = 7`, so unlike
either ladder alone this one *can* reach significance). It is reported whatever it says, and it is a
better-powered version of the same question both ladders ask.

**A diagnostic, not a band, reported either way:** the seed-only **strength** span. If changing only
the seed moves sampled `k=-1` materially, then the recipe does not control memoriser strength
either, and "strength" and "seed" are not separable even here. That would be a finding about the
recipe and is reported as one rather than treated as a failure of the arm.

**Entry gate, unchanged.** A point enters only if its sampled `k=-1` recall is at least `0.10`
(caution (a)). A point below it is excluded and reported as excluded.

## What each outcome licenses, written down before the numbers exist

- **No outcome of this arm restores the retracted sentence.** It was retracted in feat-120 because
  the ordering inverted on a third corpus, and nothing about seeds can un-invert that.
- If it is **run-to-run variation**, the consequence is larger than feat-121's: every onset ratio in
  the paper is a single draw from a distribution the paper never sampled, and the nine-pair table
  must be presented as nine draws rather than nine measurements. Section 4's table keeps its role as
  evidence that extraction begins near `s(x)` and loses any per-pair reading, which is already where
  feat-121 left it --- this would make that demotion final rather than precautionary.
- If it is **strength**, the nine-pair table is confounded by memoriser strength specifically, which
  is at least measurable: every pair's sampled `k=-1` is already on disk, so the table can carry a
  strength column and a reader can see it.
- If **inconclusive**, the paper says the two cannot be separated at this sample size and reports
  both ladders' spreads side by side.

**What will not happen.** No second seed set, no fifth seed, no re-threshold, no re-grid, no
swapping the epoch count after seeing the result, and no dropping a seed that lands awkwardly.

## Scoring log

## Scoring, 2026-09-16 --- RUN-TO-RUN VARIATION. The band that costs us the most fired.

Command, and its output is `results/strength_ladder_seeds.csv`:

```
.venv/bin/python analysis/strength_ladder.py --axis seeds
```

| point | sampled `k=-1` | `k=0` | onset | bracket | ratio |
|---|---|---|---|---|---|
| seed=1 | 0.1581 | 0.000 | 3.215 | (3.2, 3.6] | 1.0549 |
| seed=2 | 0.2045 | 0.000 | 3.239 | (3.2, 3.6] | 1.0626 |
| seed=0 (feat-120) | 0.1504 | 0.000 | 4.007 | (3.6, 4.2] | 1.3146 |

All three clear the entry gate, so none is excluded. `k=0` is `0.000` at every point.

**Seed-only ratio span `0.2597` (1.0549 to 1.3146), at or above the committed `0.20`.** Against
feat-121's epoch-only `0.4721` on the identical pair, corpus, anchor and grid, the seed alone
reproduces **55%** of what four different epoch counts produced.

**Seed 3 is still fine-tuning and cannot change this verdict.** The committed quantity is the span
over the four points, and a span is a maximum minus a minimum: adding a fourth point to a set can
only hold it or widen it, never narrow it. `0.2597` is therefore a lower bound on the four-point
span, and the four-point span is already `>= 0.20` whatever seed 3 returns. Its number is appended
below when it lands, and the committed **secondary** — the seven-point pooled Spearman — waits for
it, because seven points is what was registered and six is not seven.

**The committed diagnostic fired too, and it is the reason the arm reads this way.** The seed-only
*strength* span is **1.36x** (0.1504 to 0.2045). The pre-registration said in advance that if the
seed moves sampled `k=-1` materially then "strength" and "seed" are not separable even here, and it
does: within this ladder `rho(sampled k=-1, ratio) = -0.500`, the same sign as feat-121, with the
strongest memoriser carrying the lowest ratio. **This arm did not isolate the seed. It re-ran a
strength ladder with the seed as the knob that moved strength.**

### What the other three ladders say, and why this one is different

This is the fourth seed ladder, not the only one. Read together they do not say "seeds do not move
the onset ratio", which is what a reading of the first three alone supported and what was briefly
written into the manuscript before this arm landed:

| ladder | memoriser, sampled `k=-1` | strength span | ratio span |
|---|---|---|---|
| KL3M-520M, CopyBench | 0.5149 – 0.5756 | 1.12x | 0.0333 |
| KL3M-520M, BookMIA | 0.6690 – 0.7326 | 1.10x | 0.0663 |
| Pleias-1.2B, CopyBench | 0.9091 – 0.9615 | 1.06x | 0.0795 |
| **Pleias-1.2B, BookMIA** | **0.1504 – 0.2045** | **1.36x** | **0.2597** |

The three reproducible ladders are the three whose memoriser recovers half the passage or more. The
one that is not reproducible is the one whose memoriser **barely clears this paper's own entry gate
of 0.10** (caution (a)), and it is also the only one whose seed moves strength by more than 1.12x.
The ordering is monotone in memoriser strength across all four.

**So the finding is not about seeds.** It is that the onset ratio is a function of memoriser strength
and inherits that function's noise: where the memoriser is saturated, re-seeding lands within
`0.033`–`0.080`; where it is marginal, re-seeding moves strength by a third and the ratio by `0.26`.
Nothing here is evidence that the seed has an effect of its own.

### What this licenses, per the contract written above

The retracted sentence stays retracted; no outcome of this arm could have restored it. Of the three
consequences written down in advance, the **run-to-run** one is the one that fired, and its stated
price is paid in full: the nine-pair table is presented as evidence that extraction begins near
`s(x)` and carries **no per-pair reading at all**. The stronger form written into the manuscript an
hour before this arm landed — that the ratio "is reproducible for a given pair and a given memoriser
recipe" — is **withdrawn as stated and replaced** by the strength-conditioned version above, which
is what all four ladders support.

**A consequence about our own protocol, not in any band, reported because the arm surfaced it.** The
entry gate of sampled `k=-1 >= 0.10` admits pairs whose onset ratio is a single draw with a spread
of `0.26`. It was set (caution (a)) to keep out pairs whose greedy recall lies, and it does that;
it was never validated as a floor for a *stable* onset estimate, and this arm shows it is not one.
Nothing is re-gated retroactively — that would be choosing the answer — but the limitation is now
stated in the paper.

**Not done:** no second seed set, no fifth seed, no re-threshold, no re-grid, no swapped epoch
count, and no seed dropped. Seed 3 is running and will be appended whatever it says.

### Seed 3 landed 2026-09-16 08:18, and the committed secondary with it

```
.venv/bin/python analysis/strength_ladder.py --axis seeds    # four points now
.venv/bin/python analysis/strength_ladder.py --axis pooled   # the committed secondary, seven points
```

| point | sampled `k=-1` | `k=0` | onset | bracket | ratio |
|---|---|---|---|---|---|
| seed=1 | 0.1581 | 0.000 | 3.215 | (3.2, 3.6] | 1.0549 |
| seed=2 | 0.2045 | 0.000 | 3.239 | (3.2, 3.6] | 1.0626 |
| **seed=3** | **0.2391** | 0.000 | 2.950 | (2.9, 3.2] | **0.9677** |
| seed=0 (feat-120) | 0.1504 | 0.000 | 4.007 | (3.6, 4.2] | 1.3146 |

**The committed four-point span is `0.3469`** (0.9677 to 1.3146), against the committed threshold of
`0.20`. The verdict written above was `0.2597` on three points and is unchanged in kind and stronger
in degree, which is what the monotone argument said would happen: a span is a maximum minus a
minimum, so the fourth point could only hold it or widen it, and it widened it.

Seed 3 is the strongest of the four memorisers (`0.2391`) and carries the lowest ratio (`0.9677`),
the same direction the other three points show. Its fine-tune ran all `40` epochs at a final loss of
`0.0619` against the `0.02` stop-loss — **it did not converge, exactly like seeds 0, 1 and 2 on this
cell**, so the convergence reading offered below is unchanged by it.

**Which span belongs where, because the two are not interchangeable.** The four-point `0.3469` is
this arm's own committed primary and is the number to read against feat-121's epoch ladder, since
that ladder also has four points — four seeds against four epoch counts is like for like, `0.3469`
against `0.4721`, so the seed alone reproduces **73%** of what changing the training length did. The
three-point `0.2597` is the number `onset_prediction_seedspread2.md` compares KL3M against, because
KL3M has three seeds and a span grows with the number of draws. Neither is quoted in the other's
place.

### The committed secondary, scored: seven points, exact `p`

Pooling feat-121's four epoch points with this arm's three new seeds gives the seven distinct
(memoriser, ratio) points registered in advance, sharing the `seed 0`, `40`-epoch corner:

```
rho(sampled k=-1, onset ratio) = -0.714, exact p = 0.0881   (n = 7, floor 1/2520)
```

Reported as registered, whatever it says: **it does not reach significance.** The sign is the one
every ladder in this project shows — a stronger memoriser carries a lower onset ratio — and at seven
points that is a direction, not a demonstration. It is the better-powered version of the question
both ladders ask and it answers it the same way the underpowered versions did: consistent in
direction, unresolved in magnitude.

**A protocol incident on this arm, recorded because the reader of a scoring log should know.** The
queue shell running these three seeds was started at 02:57 and `scripts/run_strength_ladder.sh` was
edited and committed at 03:17, while it was executing. Bash reads a script incrementally by byte
offset, so the edit shifted every later offset; the loop body had already been parsed and all three
iterations ran the original code, but when the shell returned to the file after the loop it landed
mid-command and reported `line 52: --base: command not found` followed by a spurious
`FAILED finetune seeds=3`. **The failure is after all work completed and no measurement is affected**,
which was checked rather than assumed: all four points carry the identical 16-point grid, `n=100`,
`k=0` exactly `0.000`, zero invariant violations, and identical `lr`, `rank`, `batch`, `accum`,
`stop-loss` and base model in their recipes. Never edit a shell script while it is running.
