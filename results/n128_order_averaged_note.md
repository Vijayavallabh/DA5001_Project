# Is SATURATED BY 64 an artefact of presentation order? — a POST-HOC check, with no committed bands

This is **not** a pre-registration and there are **no committed bands**. The registered band of
`results/onset_prediction_n256.md` Arm A has already been read and scored (`+0.0140`
`[-0.0180, +0.0460]`, **SATURATED BY 64**), and nothing here can change that reading. It is
deliberately **not** named `onset_prediction_*`, for the reason `scorer_free_cost_note.md` and
feat-115 were not: a post-hoc re-analysis must not be able to inflate the pre-registration count.

## Why it exists

Scoring Arm A turned up a property of the instrument that was not previously recorded (caution
`(ap)`): **the judged level of an arm depends on which other arms are in the sweep.**
`selection_scaling.py` draws one `rng.random()` per item of
`distinct = sorted({(p, picks[(p,n)]) for p in pids for n in grid})` to fix presentation order, and
the eighth arm grew that set from `1,954` to `2,219` — so nearly every shared item was shown in the
opposite order to the committed pass, into a judge caution `(m)` measured as position-dominated
(`261` of `500` win shown second, `24` shown first). Judge B scored the `n=1` arm at `0.435` in the
committed pass and `0.478` here, on text that is **byte-for-byte identical** (`32,000` of `32,000`
rewards at ranks `0–63` bit-identical).

The registered band is immune to this by construction — it is a **paired difference within one
pass**, so both arms share the flip sequence. But "immune in expectation" is not "measured", and the
paper's own position-free construction is order averaging: judge each pair in **both** orders and
average, which removes position by construction rather than in expectation (caution `(m)`,
`results/order_averaged_h2h.csv`).

## What is run

`analysis/order_averaged_h2h.py` at `--n 64` and `--n 128`, both against the **same** pool
(`output/phase5/sel_anchor128`, `results/selection_rewards128.csv`) and the same opponent, so the
only difference between them is `n`. Tags `_n128oa64` and `_n128oa128`. GPU 2, about `0.07` gpu-h
each on the record of the nine `h2h_*` runs already in `compute_hours.csv`.

## What it can and cannot say

It **cannot** re-open the registered reading, and it is not offered as a substitute for it. It can
say whether the same conclusion survives the construction the paper trusts most. Three outcomes and
what each would mean, written down before the numbers exist:

- The order-averaged `g(128) - g(64)` also has an interval containing zero → the registered reading
  stands under both constructions, and the saturation is not a position artefact.
- It is clearly positive → the single-order read was conservative; the registered verdict still
  stands as the committed one, and this is reported beside it as a disagreement between
  constructions, not as a correction of the band.
- It is clearly negative → same, in the other direction.

In every case the manuscript keeps the registered verdict and reports this beside it.

## Scoring

*(not yet run)*
