# Post-hoc note: the meter run as speculative decoding (review 2 Q3)

**Post hoc, 2026-09-25, descriptive, no band.** Review 2 (Q3): "You concede you did not try the
speculative-decoding meter, a cheap control that could erase the batched-time advantage; why is it absent?"
Section 4 prices selection at one request per call at `0.220x` the `70`B meter's time and `0.514x` the `8`B
meter's (`results/batched_latency.csv`).

**What a speculative meter is.** The meter serves `p*_t = normalize(p_s^{bc_t} p_r^{bd_t})`, and its per-step
log records `bc_t` and `bd_t`. It can be served by exact speculative sampling with the anchor as the draft. A
drafted token is kept with probability `alpha_t = sum_x min(p_s(x), p*_t(x)) = 1 - TV(p_s, p*_t)`, and the
served law is unchanged.

**What was measured** (`analysis/speculative_acceptance.py`, host B, one H100, run twice with identical
output). Both models were teacher-forced on the committed `8`B meter's own served tokens (`output/phase2/conc_all`,
lowest seed, `500` prompts), and `p*_t` was rebuilt from the logged weights. The rebuild reproduces the logged
`p*` of the served token to a median `0.0002`. Mean acceptance:

| `k` | steps | mean `alpha` | median | 10th pct. |
|---|---|---|---|---|
| 10 | 99,907 | 0.599 | 0.592 | 0.337 |
| 3 | 95,158 | 0.599 | 0.591 | 0.337 |
| 0.5 | 75,700 | 0.681 | 0.640 | 0.552 |

**What it buys, on a cost model optimistic for the meter.** A round that drafts `g` tokens is charged `g`
anchor decode steps plus ONE risky decode step for verifying `g+1` positions. That charge is exact only where
decoding is memory-bound. It also ignores the meter's own solve. Per-token costs come from
`results/batched_latency.csv` at one request per call: the anchor alone `0.01186` s, the `8`B alone `0.01597`
s and the meter `0.03002` s. The `70`B pair's are `0.01280`, `0.06095` and `0.07419` s. The tokens served per
risky forward are a Monte Carlo over each trajectory's own `alpha_t`, and the helper matches the closed form
`(1 - alpha^{g+1})/(1 - alpha)` at constant `alpha`.

- **`8`B pair.** The best round is `g = 1`: `1.59` tokens per risky forward, a meter `1.71x` faster, and
  selection at `n=64` at **`0.880x`** the meter's time instead of `0.514x` (`k = 10`). At `k = 3` the figure is
  `0.881x`, and at `k = 0.5` it is `0.928x`. Parity would need a constant acceptance of `0.802`. At perfect
  acceptance selection would take `1.254x` the meter's time.
- **`70`B pair.** Measured the same way on `output/phase5/imit_llama70b`, two H100s, once cards freed up.
  Acceptance is `0.615` at `k = 20` and `0.618` at `k = 3` (the timed meter ran at `k = 10`). The rebuilt
  `p*` matches the logged probability to a median `0.0005`. The best round is `g = 2`: `1.96` tokens per
  risky forward, a meter `1.68x` faster, and selection at **`0.370x`** the meter's time instead of `0.220x`
  (`0.372x` at `k = 3`). Even at **perfect** acceptance the best round with `g <= 8` leaves selection at
  `0.901x` (`results/speculative_acceptance_70b.csv`, no model needed).

So a speculative meter narrows selection's per-request advantage, from `0.51x` to at most `0.88x` at the `8`B
pair and from `0.22x` to at most `0.37x` at the `70`B (`0.90x` even at perfect acceptance), but does not
erase it at either. These are bounds on a
favourable cost model, not timings. They bound latency at one request per call only. The throughput price (`21.8x`, measured at a batch width of `200`)
is not re-estimated. At that width decoding is closer to compute-bound, so verifying `g+1` positions costs more
than one decode step, and the optimistic charge used here would not hold.
