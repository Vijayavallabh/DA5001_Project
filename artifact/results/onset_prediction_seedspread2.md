# Does the within-pair seed spread generalise, or is it Pleias-1.2B's alone?

**Committed 2026-09-16 03:15, before either of these memorisers exists**, and while the
Pleias-1.2B seed arm it will be compared against is still on its first of three sweeps, so no
number from that arm exists either. Nothing above the scoring-log heading is edited afterwards.

## Why a second pair, and why this one

feat-121 measured a `0.4721` onset-ratio spread within Pleias-1.2B on BookMIA from varying
`--epochs` alone, against the nine-pair table's whole between-pair spread of `0.2874`. The paper now
says, on that basis, that the nine-pair ratios are not readable as per-pair properties. **That
sentence rests on one pair.** If Pleias-1.2B is peculiar --- it is the weakest memoriser of the
three, at sampled `k=-1` of `0.1504` where the others read `0.7326` and `0.4425` --- then the
generalisation is too strong and the paper is overclaiming in the direction of its own retraction,
which is still overclaiming.

KL3M-520M is the right second pair and not a convenient one. It is the **fine-tokenizer** pair, the
one whose interval excluded `1` on CopyBench and Gutenberg and whose behaviour the whole tokenizer
split was built on; it is the **strongest** BookMIA memoriser, so it tests the spread where the
memoriser is least likely to be the loose variable; and it is the pair whose result the paper would
most like to keep. If its seed spread is small, the claim we just wrote is too broad.

## The design

Identical to the Pleias seed arm in every respect except the pair: `--epochs 40` fixed (the recipe
every published memoriser here uses, and the one its own `seed 0` run used), varying `--seed` alone.
Two new seeds, `1` and `2`, one per card, run in parallel because two cards are free for a bounded
window; `seed 0` is its existing BookMIA run, already measured and bootstrapped.

The grid is **KL3M-520M's own committed BookMIA grid**, `-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2
3.6 4.2`, not the extended one --- because that is the grid its `seed 0` point was measured on, its
no-crossing fraction there was `0.0%`, and its onset at `2.465` sits far from either end. Matching
the corner's grid is what makes the four points one ladder.

## Committed bands, and the like-for-like rule that matters more than they do

**A three-point span is not a four-point span.** The maximum minus the minimum grows with the number
of draws, so comparing KL3M's three seeds against Pleias' four would favour the conclusion we are
testing for. The primary comparison is therefore fixed now:

> KL3M-520M's seed span over `{0, 1, 2}` against **Pleias-1.2B's seed span over the same three
> seeds `{0, 1, 2}`**, both three-point spans, both at `--epochs 40`, both on BookMIA.

Pleias' four-point span is reported beside it and is never the number KL3M is compared against.

| outcome | reading |
|---|---|
| KL3M's 3-seed span is **at least half** Pleias' 3-seed span | the within-pair seed spread is not peculiar to one pair. The paper's statement that the nine-pair ratios are not per-pair properties stands as written |
| KL3M's 3-seed span is **below a quarter** of Pleias' | it is substantially Pleias-1.2B's own, and the paper must say the spread was measured on the weakest memoriser in the set and may not transfer to the others. The nine-pair demotion is then supported by feat-120 alone, which is thinner |
| in between, or the two spans are both under `0.05` | **inconclusive**, reported as inconclusive, with both spans quoted |

**Entry gate, unchanged.** A point enters only if its sampled `k=-1` recall is at least `0.10`
(caution (a)). A point below it is excluded and reported as excluded.

**A diagnostic, not a band.** KL3M's `seed 0` run stopped at epoch `26` of `40` on `--stop-loss
0.02`. Whether the new seeds stop early too, and at which epoch, is reported --- if the recipe's
effective training length itself moves with the seed, then "fixed epochs" does not fix the amount of
training, which would be a finding about the recipe rather than about this pair.

## What this cannot do

Two pairs are not nine, and neither this arm nor feat-121 separates memoriser strength from
run-to-run variation; the Pleias seed arm is the only thing that can, and it is still running. This
arm answers one narrower question --- whether the spread we found in one pair exists in another ---
and is reported as answering only that.

**No outcome of this arm restores any retracted sentence.** feat-120's retraction rests on a third
corpus inverting the ordering, which nothing here touches.

**What will not happen.** No third seed for this pair unless the six-hour window that opened the
second card is explicitly extended, no re-grid, no re-threshold, no swapping to a different pair if
KL3M is unhelpful, and no dropping a seed that lands awkwardly.

## Scoring log

### Amendment, 2026-09-16 03:25, recorded before any result of this arm exists

The section above says: *"No third seed for this pair unless the six-hour window that opened the
second card is explicitly extended."* At `03:23` the user directed that the spare capacity be used
--- the cards are running at `6`--`11` GiB of `79`, so the limit was never memory --- and KL3M's
fine-tune is measured at `79` s/epoch against the `177` s this arm was budgeted at. Two further
seeds, `3` and `4`, are therefore added, co-located one per card.

This is recorded here rather than by editing the clause, because the file forbids editing anything
above its scoring log. Three facts make the addition auditable rather than convenient:

1. **No result exists.** Both KL3M fine-tunes are at epoch `4` of `40` and no sweep has begun, so
   nothing about the answer is known and the extra seeds cannot have been chosen to chase one.
2. **The committed primary is untouched.** It compares KL3M's span over seeds `{0, 1, 2}` against
   Pleias' span over the same three, and it still does. Seeds `3` and `4` enter only a **secondary**
   spread over all five points, reported beside the primary and never in place of it.
3. **Spans are not comparable across different `n`**, which is the whole reason the like-for-like
   rule exists; adding points to one pair and not the other is exactly the error that rule prevents,
   so the extra seeds are barred from the comparison by construction rather than by intention.

The entry gate, the grid, the threshold and every band above are unchanged.

