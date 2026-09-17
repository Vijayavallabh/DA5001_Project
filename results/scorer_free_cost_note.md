# Pricing the scorer-free instance — a post-hoc join, with no committed bands

This is **not** a pre-registration and there are **no committed bands**, because there is no new
run: it is arithmetic over quantities this paper already measured, joined into one table. It is
deliberately not named `onset_prediction_*`, for the reason feat-115 was not — a post-hoc
re-analysis must not be able to inflate the pre-registration count. `analysis/scorer_free_cost.py`,
`results/scorer_free_cost.csv`, no GPU.

## What prompted it

feat-114, feat-116 and feat-117 established that selection's serving cost is set by the **reward
model**, not the anchor: at `n` it runs `n(P_s + P_f)(L_p + T)` against the metered decoder's
`(P_s + P_r)(L_p + T)`, with `P_f = 7.6156` and `P_s = 1.7586`. The paper concedes `61.3x` at
`n = 64` on that basis and, after feat-118, concedes it without the saturation escape.

**Majority vote has no reward model.** Self-consistency serves the modal answer over the same `n`
anchor draws, so its "scorer" is a regex and costs no forward pass at all: `n P_s (L_p + T)`. It
carries the identical `log n` certificate, because Proposition 1 assumes nothing whatever about the
score and a mode is a score. The paper has said since v8 that self-consistency is an instance of the
mechanism. It has never said that the instance is also by far the cheapest one.

## Correction, 2026-09-18: the cost column and the accuracy columns are not the same system

Found on read-through 5. `analysis/serving_cost.P_ANCHOR` is `1.7586` --- TinyComma, the **audited**
anchor --- while every accuracy in the table below is **Comma-7B's**, because Comma-7B is the anchor
that clears the floor on GSM8K and TriviaQA. Drawing is linear in the anchor, so the cost column
prices a system that did not produce these accuracies, **in the direction that flatters the
scorer-free rule**:

| rule | `n` | cost at the audited `1.8`B | cost at Comma-7B |
|---|---|---|---|
| majority vote | 8 | `1.44x` | `3.73x` |
| majority vote | 32 | `5.75x` | `14.90x` |
| majority vote | 64 | `11.50x` | `29.81x` |
| reward `7.6`B | 32 | `30.64x` | `31.12x` |
| reward `7.6`B | 64 | `61.29x` | `62.23x` |

The reward cells barely move because the `7.6`B scorer dominates them at either anchor; the
majority-vote cells move by `2.59x` because the anchor IS their whole cost. **The ordering survives
and so does every comparison built on it** --- both rules draw from the same anchor, so the table's
ranking is unchanged --- but the *size* of the scorer-free saving is not anchor-free: at a matched
`n` majority vote costs `18.8%` of the reward rule at `1.8`B and `47.9%` at `7.0`B, `2.55x` less of
a saving. The manuscript now reports both and says which belongs with the accuracies.

## The table


Sorted by serving cost. Accuracies are feat-101's and feat-118's, unchanged; the cost column is the
model in `analysis/serving_cost.py`, unchanged.

| rule | `n` | cost | GSM8K | gain | TriviaQA | gain |
|---|---|---|---|---|---|---|
| majority vote | 2 | `0.36x` | `0.320` | `+0.0000` | `0.280` | `+0.0000` |
| majority vote | 4 | `0.72x` | `0.392` | `+0.0720` | `0.300` | `+0.0200` |
| majority vote | 8 | **`1.44x`** | `0.466` | `+0.1460` | `0.322` | `+0.0420` |
| reward `7.6`B | 2 | `1.92x` | `0.352` | `+0.0320` | `0.282` | `+0.0020` |
| majority vote | 16 | `2.87x` | `0.500` | `+0.1800` | `0.322` | `+0.0420` |
| reward `7.6`B | 4 | `3.83x` | `0.344` | `+0.0240` | `0.284` | `+0.0040` |
| majority vote | 32 | **`5.75x`** | **`0.546`** | **`+0.2260`** | `0.328` | `+0.0480` |
| reward `7.6`B | 8 | `7.66x` | `0.364` | `+0.0440` | `0.256` | `-0.0240` |
| majority vote | 64 | `11.50x` | `0.542` | `+0.2220` | `0.334` | `+0.0540` |
| reward `7.6`B | 16 | `15.32x` | `0.356` | `+0.0360` | `0.242` | `-0.0380` |
| reward `7.6`B | 32 | `30.64x` | `0.380` | `+0.0600` | `0.252` | `-0.0280` |
| reward `7.6`B | 64 | **`61.29x`** | `0.386` | `+0.0660` | `0.266` | `-0.0140` |

**Every scorer-free cell beats every reward cell on both tasks, and the cheapest one does it at a
fortieth of the price.** Majority vote at `n=8` costs `1.44x` and gains `+0.1460`; the best reward
cell costs `61.29x` and gains `+0.0660`. At `n=32` the scorer-free rule gains `3.42x` as much for
`9.4%` of the cost. On TriviaQA the reward model goes *negative* at every `n >= 8` while majority
vote rises monotonically to `+0.0540`.

The certificate is identical down the whole column: `log n`, `3.466` nats at `n=32`, against the
metered decoder's `2000` certified for the same median response.

## Three limits, which belong wherever the numbers are quoted

1. **Post hoc.** No band was committed. The accuracies come from arms that had them; this join did
   not.
2. **It needs a canonical answer.** There is no majority over free-form text, so this does not
   transfer to the judged workload where every other comparison in the paper lives. It is a claim
   about tasks with a checkable answer and nothing wider.
3. **No metered decoder was run on GSM8K.** The `x metered` column is this paper's standard cost
   denominator — the `200`-token workload of `serving_cost.py` — and **not** a measured head-to-head
   on this task. What the accuracy columns compare is each rule against the same anchor at `n=1`.

## What it must not be read as saying

Majority vote does not reach the unconstrained risky model, which scores `0.786` greedy and `0.726`
sampled on these problems against its `0.546`. Section 7 already says an anchor that cannot do the
task gains nothing at any `n`, and that still holds: this is a comparison **among mechanisms that
carry a certificate**, not a claim to have beaten the model the certificate exists to bound.

Nor does it make the reward model pointless. The judged workload is free-form, a mode does not
exist there, and every judged result in this paper — including the head-to-head the abstract stands
on — is a reward-scored arm. What changes is the compute story: `61.3x` is the price of putting a
reward model in the loop, not the price of the mechanism, and where an answer is checkable a
deployer should not pay it.
