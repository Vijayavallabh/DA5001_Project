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
