# The compute-matched comparison with the judge removed

Derived analysis over committed CSVs, not a new experiment. It has **no pre-registered bands** and
is deliberately not an `onset_prediction_*.md`: the accuracies it reads were already on record and
visible, so bands written now would be theatre (the convention `selection_argmax_stability_note.md`
follows, and caution (as) states).

Produced by `.venv/bin/python analysis/compute_matched_judgefree.py --out results` ->
`results/compute_matched_judgefree.csv`.

## Why

`results/compute_matched.csv` answers "at the meter's serving cost, which mechanism gives more
utility?" on the **judged** axis, and the answer is a loss for selection, `-0.0395 [-0.0720,
-0.0065]` at `0.92x` --- the paper's largest concession. The judge panel has since split on the
headline difference (`onset_prediction_frontier_judge.md`: two of three judges resolve it, one does
not), so a comparison resting on one judge is worth less than it was. The same question can be put
with no judge at all, on TriviaQA at TinyComma-1.8B --- the one task where both mechanisms have run
at the one anchor a meter shares a vocabulary with.

## The cost model, and which way it is biased

`analysis/serving_cost.py`'s, unchanged, with one simplification that favours the **meter**:
majority vote needs no scorer (the score is a mode over `n` strings), so selection pays
`n * P_anchor` against the meter's `P_anchor + P_risky` per served token. Token counts cancel
exactly --- both arms answer the same `500` questions at `T_max = 24`. Parity is `n = 5.57`.

**The metered decoder's serving cost does not depend on `k`.** It runs both models at every step
whatever the budget is, so its entire accuracy range is available at one cost.

## The measurement

| arm | cost vs metered | accuracy | certificate |
|---|---|---|---|
| selection `n=4` | `0.719x` | `0.118` | `1.386` nats |
| selection `n=8` | `1.437x` | `0.146` | `2.079` nats |
| selection `n=64` | `11.498x` | `0.190` | `4.159` nats |
| **metered, best arm (`k=20`)** | **`1.000x`** | **`0.618`** | **`480` certified, `44.85` realised** |

## The reading, stated against us

**The metered decoder wins the compute-matched comparison on this axis outright, and by a wide
margin.** At its own serving cost selection reaches about `0.13`; the meter reaches `0.618`. No `n`
on the grid closes that gap --- at `11.5x` the compute selection is still at `0.190`. This is the
judge-free confirmation of the concession the judged arm already carried, it is larger here than
there, and it is reported in the main text rather than an appendix.

**What the meter pays for it is the paper's thesis.** Its `0.618` arrives only at `k=20`, a
certificate of `480` nats for a `24`-token answer --- `115x` selection's `4.159` at `n=64`, and
vacuous by Proposition 2, since a `24`-token answer's total surprisal under the anchor is far below
`480`. At every budget whose certificate is **not** vacuous the meter gains nothing at all: `0.112`
at `k=0.5` (`12` nats) and `0.114` at `k=1` (`24` nats), against the anchor's own `0.112`.
Selection at `n=8` reaches `0.146` for `2.079` nats.

So the two mechanisms are not on one frontier and the paper should stop implying they are on any
single axis. **The meter buys accuracy with compute cheaply and with certificate dearly; selection
the reverse.** A deployer who has a KL budget to respect cannot spend compute to escape it --- that
is what "the meter gains nothing at any non-vacuous budget" means --- and a deployer who has
compute to burn and no certificate to honour should not be using either mechanism.

## What this does not say

It is one task at one anchor with `T_max = 24`, and the anchor is the weakest in the paper
(`0.112` on TriviaQA). It says nothing about the judged workload, where the concession stands at
`-0.0395` under judge B and has not been re-measured under the panel. The `115x` is a ratio of a
**certificate** to a **certificate**, both of which are bounds, so unlike the nats comparisons
elsewhere in the paper it is like for like.
