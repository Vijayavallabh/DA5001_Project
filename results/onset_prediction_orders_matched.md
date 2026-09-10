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

---

## Result as run: a null, and the reason it is a null invalidates half the design

`analysis/order_price.py --safe-model output/phase5/anchor_kl3m-002-520m --risky-model
output/phase5/mem_kl3m-002-520m --k 3.0 --limit 20` -> `results/order_price_kl3m_k3.csv`.

| alpha | price (ordinary) | leakage (protected) | P | R | P - R |
|---|---|---|---|---|---|
| 1 | 4266.2 nats | 9661.0 nats | 1.000 | 1.000 | +0.000 |
| 2 | 3983.8 | 9106.7 | 0.934 | 0.943 | -0.009 |
| 4 | 3606.2 | 8293.1 | 0.845 | 0.858 | -0.013 |
| 8 | 3254.0 | 7483.3 | 0.763 | 0.775 | -0.012 |

`P - R` is within 1.3% of zero at every order, which is the committed "no order dominates" band.

**The embarrassment check passes and the design still fails.** The pre-registered check was whether
this instrument orders the four arms the same way Table 1's intervention rate does; it does, both
monotone in alpha. But the check was aimed at the wrong risk. The real defect is that the two
columns are not the same kind of quantity, and the paper says so itself two sections earlier:
*"Extraction and utility differ because one is a rare event and the other a bounded average."*

Fidelity `-D(p_r || p_theta)` is an average, so it is the right instrument for price and the wrong
one for leakage. Verbatim reproduction is the probability of a long run of exact tokens, a product
over hundreds of steps; a 15% reduction in average fidelity can collapse that product by orders of
magnitude and this measurement cannot see it. That is exactly why Table 1's oracle recall falls
$24\times$ from alpha = 1 to alpha = 4 while the table above moves by 14%. The two are not in
conflict; they are measuring different functionals, and only one of them is leakage.

So the null is real for what was measured and the measurement was half wrong. Withdrawn as a
dominance test.

## The corrected design, committed before it runs

Keep the price column -- fidelity on ordinary generations is a bounded average and the right
instrument for it. Replace the leakage column with the quantity whose exponential *is* the
reproduction probability:

    L(alpha) = sum_t log p_theta(x_t | x_{<t})    over the protected token sequence

evaluated at the same published `k` under each order's own charge. This is deterministic, needs no
sampling, and is the rare-event functional rather than an average. The reproduction probability
ratio between two orders is then `exp(L(alpha) - L(1))`, directly comparable to the recall ratios in
Table 1.

| outcome | reading |
|---|---|
| `L(1) - L(alpha)` is large in nats while `P(alpha)` stays near 1 | the published `k` conceals a strictly better operating point: an order that costs little of what the budget buys and collapses reproduction. This is the matched-utility statement Appendix D concedes it lacks |
| both move together in proportion | no dominance; the concession stands as written |
| `L` moves less than `P` | raising the order costs more utility than reproduction it prevents |

**Sanity condition, recorded now:** `L(1)` must be far *below* the unconstrained model's own
log-probability of the same tokens and far *above* the anchor's, or the instrument is not measuring
a constrained decoder at all. Both bounds are computed and reported alongside.

## The sanity condition fired on the first run, and it found a split bug

The corrected instrument ran on KL3M-520M at k = 3, 25 passages a side, and the bracket it was
required to print came out **inverted**:

```
the bracket the constrained decoder must sit inside:
  risky model assigns -37085.9 nats to the protected tokens, the anchor alone -23023.0
```

The memoriser finds the "protected" passage `e^{14063}` times *less* likely than the clean anchor
does. No constrained decoder sits inside an inverted bracket, so the instrument was measuring
something other than what it claimed, and the recorded condition -- `L(1)` far below the risky
model's own log-probability and far above the anchor's -- is what surfaced it in one run.

**The cause.** `analysis/order_price.py` selected `split == "test"`, and every phase-5 memoriser is
fine-tuned on `attack_train` + `val` with `test` held out (`output/phase5/mem_*/recipe.json`,
`"splits": ["attack_train", "val"]`). The two splits are also disjoint in *novel*: `attack_train`
carries `a_game_of_thrones`, `casino_royale`, `dune`, `fahrenheit_451` and others the model was
trained on; `test` carries `fifty_shades_of_grey`, `harry_potter_and_the_sorcerer's_stone` and
`lord_of_the_flies`, which it has never seen. So the "protected passage" arm was a **held-out
novel**, and a LoRA-memorised model is *worse* than its own base on prose it did not memorise --
which is precisely the inversion above.

**This is not confined to feat-072.** `analysis/marginal_price.py` takes `--split` with default
`test` and `scripts/run_marginal_price.sh` does not override it, so the feat-070 "protected passage"
column -- which is in the manuscript -- is also a held-out novel rather than a memorised one. The
measurement is real and the arithmetic is right; the label is wrong, and the trajectory an
extraction adversary actually walks is the memorised one, where `l = log p_r - log p_s` is far
larger and the geometry need not be the same.

**Committed before the re-runs:** both probes are re-run on `attack_train`, the split the memoriser
was trained on, and the feat-070 table gains a third target type rather than losing one. The bands:

| outcome | reading |
|---|---|
| the memorised arm's `gain_ratio` and `frac_of_ceiling` land inside the held-out arm's [min, max] at every k | the headroom really is a property of the geometry and independent of what is being copied; the manuscript claim strengthens, with the label corrected |
| the memorised arm's headroom is materially larger (`gain_ratio` above 1.25 at any k <= 1) | greedy scheduling *does* leave room on the trajectory that matters, the Section 2 measurement is wrong as written, and the dual decoder becomes worth building |
| `frac_of_ceiling` at k = 1 falls below 0.5 on the memorised arm | the unconditional ceiling argument does not transfer to memorised text and must be restated for held-out text only |

The order-price bracket is re-checked the same way: `L(1)` must now sit strictly between the two,
or the pair is not a memoriser on these passages either and nothing further is claimed from it.
