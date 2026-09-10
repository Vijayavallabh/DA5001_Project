# Pre-registration: do the Renyi orders differ at matched utility, measured without a judge?

Committed before the runs.

## The gap this closes

Table~1 shows four decoders at one published budget spanning two orders of magnitude in what an
adversary recovers and more than two in what they cost. Appendix D concedes the limit of that
comparison in its own words: it is "at matched *budget*, not at matched utility", and Section 5
shows why the obvious fix fails --- re-judging the same generations with the same judge inverts the
ranking of the four orders, so no judged separation at n = 150 survives a re-run.

feat-070 supplies an instrument the judge cannot match. On the geodesic the fidelity
`G(theta) = theta m - psi(theta) = -D(p_r || p_theta) + const` is what a metered decoder is buying
-- it has no other signal -- and it is deterministic, needs no sampling, and is computed to machine
precision from two teacher-forced forward passes. So the price of an order can be read off directly
instead of being estimated through a noisy judge.

## The measurement

For each order `alpha` in {1, 2, 4, 8} at a fixed published `k`, solve for `theta` under that
order's charge `D_alpha(p_theta || p_s) <= k` (a_patch/renyi.py, the same solver the decoder uses)
and report, on the same passages:

* **price**   fidelity bought on *ordinary* generations -- what the deployer loses,
* **leakage** fidelity bought on *protected* passages  -- what the rights-holder loses.

Both in nats, both as a fraction of the `theta = 1` ceiling, at one published budget.

## Bands, fixed now

Write `R(alpha) = leakage(alpha)/leakage(1)` and `P(alpha) = price(alpha)/price(1)`.

| outcome | reading |
|---|---|
| `R(alpha) < P(alpha)` by a clear margin at some alpha | at the same published `k` an order exists that gives up less utility than it gives up leakage. The budget hides a strictly better operating point, and Table 1's claim holds at matched utility, not merely at matched budget |
| `R(alpha) ~= P(alpha)` for every alpha | the orders trade the two off at a constant rate; the published budget is uninformative about *which* decoder you have, which Table 1 already shows, but no order dominates and the concession in Appendix D stands as written |
| `R(alpha) > P(alpha)` | raising the order costs more utility than the leakage it prevents, and the paper should say so plainly rather than presenting the family as a menu |

**The prediction that would embarrass this.** Fidelity to `p_r` is not utility, and a referee is
entitled to say so. If the ordering of the four arms by this instrument *disagrees* with the
ordering by intervention rate in Table 1 (risky-unchanged 94.0%, 91.4%, 8.7%, 0.07%), then the
instrument is measuring something other than what the decoder does and the result is withdrawn, not
reinterpreted. Recorded here so that check cannot be skipped afterwards.

## What this cannot establish

It cannot establish that a deployer would *prefer* a higher order: fidelity to the risky model is
the mechanism's own objective, not a user's satisfaction, and Section 5's judged arms remain the
only measurement of the latter. What it can establish is whether the published budget conceals a
dominance relation among decoders that all publish the same number.
