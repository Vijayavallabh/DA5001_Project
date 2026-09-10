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

## Result on the memorised split, and the next question it forces

`results/order_price_kl3m_k3.csv`, KL3M-520M, `k = 3.0`, 25 memorised passages (9,500 tokens) and
25 ordinary generations. The bracket now holds with room to spare -- the memoriser assigns
$-26.1$ nats to the passages it was fine-tuned on and the anchor alone $-23{,}024.6$, and every
constrained arm sits strictly between them -- so the instrument is measuring a constrained decoder.

```
 alpha   price      P    log p/token   nats/window   window x less likely
     1   5262.0  1.000       -0.1854          0.00                     1
     2   4951.6  0.941       -0.5001         15.74               6.84e+06
     4   4509.2  0.857       -0.7332         27.39               7.85e+11
     8   4082.2  0.776       -0.8556         33.51               3.59e+14
```

Quoted per 50-token window because that is the unit Table 1's recall is scored in. **This is the
first band: `P` stays near 1 while `L` collapses.** Raising the order from 1 to 4 at the *same*
published budget costs $14\%$ of the fidelity the budget buys and makes an exact 50-token window
$8 \times 10^{11}$ times less likely. The exponent is much larger than Table 1's measured $24\times$
drop in oracle recall, and it should be: recall is near-verbatim with retries and a similarity
threshold, this is exact reproduction of every token. They agree in direction and the rare-event
functional is far the more sensitive of the two.

**What this does not yet say.** It is dominance at matched *budget*, which is the axis Appendix D
already has. The statement that would close the concession is dominance at matched *utility*: give
`alpha = 4` whatever budget it needs to buy the same fidelity `alpha = 1` buys at the published `k`,
and ask what it leaks there. Fidelity is monotone in `k` at fixed order, so that budget exists and
is unique.

### Pre-registration, committed before the run

Sweep a grid of `k` per order in one pass over the same passages (no new forward passes, only more
theta solves), interpolate the `k'(alpha)` at which the ordinary-side fidelity equals `alpha = 1`'s
at the published `k`, and report the protected-side `log p(target)` there.

| outcome | reading |
|---|---|
| at matched fidelity, `L(alpha) - L(1)` is still large and negative | the order dominates on the axis that matters: same utility to the deployer, orders of magnitude less reproduction. Appendix D's concession is closed and the published KL decoder is off the efficient frontier |
| at matched fidelity the two agree to within a window factor of $10$ | the order buys nothing a budget increase would not; the four arms trace one frontier and Table 1's ranking is an artefact of comparing at matched budget |
| `k'(alpha)` exceeds the vacuity threshold `s(x)` for the pair | the matched-utility point is outside the region where the certificate says anything, and the comparison is moot for a deployer |

The instrument is only read if the bracket holds at every cell: `logp_target_safe < L < logp_target_risky`.

## The replication, and what it does not settle

`results/order_price_pleias_k3.csv`, Pleias-1.2B, same protocol, 6,805 protected tokens. The
bracket holds again -- memoriser $-8.3$ nats, anchor $-21{,}222.6$ -- and the two pairs agree
closely:

```
                 KL3M-520M                    Pleias-1.2B
 alpha    P    window x less likely      P    window x less likely
     2  0.941            6.84e+06      0.941            8.37e+07
     4  0.857            7.85e+11      0.884            6.89e+13
     8  0.776            3.59e+14      0.843            9.13e+16
```

`P` at `alpha = 2` is $0.941$ on both pairs to three decimals.

But a smoke run of the matched-utility sweep on three passages already warns that this is the
weaker of the two statements. At the published $k = 3$ the audited decoder already buys $98.5\%$ of
the unconstrained ceiling, so matching its fidelity pushes the higher orders to $k \approx 8$, where
none of them binds and all four serve the same distribution. The matched-budget dominance above may
therefore be an artefact of comparing at a budget where `alpha = 1` is nearly unconstrained and the
others are not -- which is the second band, and it would make Table 1's ranking a statement about
the comparison rather than about the decoders. The full sweep, on 25 passages and a 12-point grid on
both pairs, is what decides it, and it is running against the bands committed above.

## Scored: the matched-utility sweep, 25 passages and a 12-point grid on both pairs

`results/order_frontier_{kl3m,pleias}{,_matched}.csv`. The bracket holds at every one of the 48
cells per pair and nothing saturates, so the grid is entirely inside the constrained region.

```
                     KL3M-520M  s(x)=2.415              Pleias-1.2B  s(x)=3.209
published k  alpha   k matched   window x less likely    k matched   window x less likely
        1.0      2       1.963                    871        1.643               3.54e+04
        1.0      4       3.105                    467        2.264               1.68e+07
        1.0      8       4.008                   1.16        2.663               3.32e+07
        3.0      2       5.450                      1        4.733                   52.2
        3.0      4       7.058       0.07 (leaks MORE)        5.483                     82
        3.0      8       7.986       0.01 (leaks MORE)        6.164                   21.9
```

**All three bands fire, in different cells, and the pattern is the finding.**

*Band 3 fires at the published $k = 3$ on both pairs.* Matching what `alpha = 1` buys there costs
between $4.7$ and $8.0$ nats per token, every one of them past that pair's vacuity threshold
$s(x)$. The comparison Table 1 draws at $k = 3$ is between operating points at which the certificate
already says nothing. It is also nearly degenerate on its own terms: at $k = 3$ the audited decoder
captures $98.3\%$ (KL3M) and $97.2\%$ (Pleias) of the unconstrained ceiling, so matching it forces
the higher orders to budgets where they do not bind either, and all four serve very nearly the same
distribution.

*Band 2 fires at $k = 3$ on KL3M-520M.* `alpha = 2` is a wash and `alpha = 4` and `8` leak
**more** at equal utility. On this pair, at this budget, the matched-budget ranking in Table 1
reverses once the deployer is given back what the order took from them.

*Band 1 fires at $k = 1$ on both pairs*, which is the only cell where the constraint genuinely binds
($83.6\%$ and $78.9\%$ of ceiling) and the matched budgets stay near or below $s(x)$. There the
higher order does dominate: at identical fidelity, `alpha = 2` makes an exact 50-token window
$871\times$ (KL3M) and $3.5\times10^4$ (Pleias) less likely. But the size of the effect spans
**four orders of magnitude between two pairs**, and it is **non-monotone in `alpha`** -- on
KL3M-520M `alpha = 8` gives back the entire advantage ($1.16\times$), and on Pleias `alpha = 8`
adds nothing over `alpha = 4`.

**The reading.** Appendix D's concession is closed, and not by the answer the matched-budget table
suggested. A higher Renyi order is not uniformly a better decoder; at matched utility it is worth
between $10^7$ and *less than one* depending on the pair, the budget and the order. What Table 1
ranks is the charge function, not the decoder --- which is this paper's thesis one level up: the
same published $k$ is not comparable across charges any more than it is across works. The
matched-budget dominance recorded in the previous section is real arithmetic and the wrong
comparison, and it is reported that way.

## What predicts the four-orders-of-magnitude spread? Pre-registered before the re-analysis

The matched-utility advantage at `k = 1` is $871\times$ on KL3M-520M and $3.5\times10^{4}$ to
$3.3\times10^{7}$ on Pleias-1.2B, and nothing in the paper predicts which pair gets which. One
mechanism is already visible in the `k = 3` row group: the advantage vanishes exactly where the
audited decoder stops being constrained, because matching a nearly unconstrained decoder forces
every other order to a budget where it does not bind either. That suggests a single explanatory
variable, and it is one the deployer can compute:

    F(k) = the fraction of the theta = 1 ceiling the audited decoder captures at k

`F -> 1` must force the advantage to 1, which is an anchor point the hypothesis cannot dodge. The
existing grids already contain this at 12 budgets per pair, so the test is a re-analysis with no new
compute: for every grid `k` treated as the published one, compute the matched budget and the window
factor for each order, and plot `log10` of that factor against `F(k)`.

| outcome | reading |
|---|---|
| the two pairs' curves lie within one order of magnitude of each other at matched `F`, and both go to $1$ as `F \to 1` | the order's value is predicted by how far the audited decoder is from saturation, and a deployer can compute it from the anchor and the risky model alone. This is a law, and it gets its own measurement |
| the curves are separated by more than an order of magnitude at matched `F` | `F` is not the variable; the advantage is pair-specific, which strengthens the paper's thesis and is reported as a negative |
| the curves are not monotone in `F` | the framing is wrong and the re-analysis is reported without a fitted law |

Scored on both pairs at once; the two new pairs (Phi-3.5-mini, Comma-7B) are run only if the first
outcome holds, as an out-of-sample test rather than as more fitting data.

## Scored: saturation does not predict it

`results/order_law{,_summary}.csv`, from `analysis/order_law.py`, a re-analysis of the two grids
with no new compute. Every grid `k` is treated in turn as the published budget, and the curves are
interpolated onto a common `F` grid because the two pairs' budgets do not land on the same `F`.

```
   F     alpha=2  alpha=4  alpha=8      <- decades between the two pairs at matched F
0.70        1.36     3.33     6.06
0.80        1.42     3.72     6.46
0.90        1.44     3.80     5.67
0.95        1.50     3.17     4.60
0.99        1.15     1.66     2.08
```

**Outcome 2, cleanly: `F` is not the variable.** At every level of saturation the two pairs differ
by more than a decade -- a median of $3.11$ and up to $6.46$ -- so the fraction of the ceiling the
audited decoder has captured does not tell a deployer what a higher order is worth. The anchor
point survives in magnitude but not in sign: at $F \ge 0.99$ the largest factor is $10^{1.38}$, and
the two pairs sit on *opposite sides of one*, KL3M-520M leaking $24\times$ **more** at $\alpha=8$
while Pleias-1.2B still leaks $10^{0.57}$ less.

One partial regularity did appear and is recorded as a hypothesis, not a result: at $\alpha = 2$ the
offset between the pairs is $+1.37$ decades with a standard deviation of $0.11$ over $F$ from
$0.70$ to $0.99$ -- close to a constant multiplicative factor per pair -- while at $\alpha = 4$ it
is $+3.19 \pm 0.75$ and at $\alpha = 8$ $+5.06 \pm 1.67$. Two pairs cannot fit a predictor for an
offset, and fitting one on two points and then quoting it would be the mistake this file exists to
prevent.

### Pre-registered before the two remaining pairs run

Phi-3.5-mini and Comma-7B have memorisers on the same split (`output/phase5/mem_phi35mini`,
`output/phase4/memorizing_comma7b`, both `["attack_train", "val"]`), so the same grid runs on them
unchanged. This is a test of the negative, not a search for a law.

| outcome | reading |
|---|---|
| four pairs span more than two decades at matched `F`, and at least one more pair shows a sign flip at some `alpha` | the claim is that nothing a deployer publishes or can compute --- the budget, the intervention rate, the order, or the distance from saturation --- determines what the order is worth. Reported as the extension of Section 6's title |
| the two new pairs land inside the interval the first two span, and the $\alpha = 2$ offset is ordered by some measured pair property ($s(x)$, $s_r/s_s$, anchor size) | a partial law at $\alpha = 2$, reported as such and only at $\alpha = 2$ |
| all four collapse to within a decade at matched `F` | the two-pair separation was an artefact of those two pairs and the whole subsection is withdrawn |

The bracket gates every cell as before, and any pair whose memoriser does not clear it is excluded
with its numbers reported, not silently dropped.

## Phi-3.5-mini lands, and one ordering appears. Its test is committed before Comma-7B runs.

`results/order_frontier_phi{,_matched}.csv`. Bracket holds at all 48 cells (memoriser $-184.4$
nats, anchor $-19{,}695.3$). At $k=1$: $\alpha=2$ buys $133\times$, $\alpha=4$ $67\times$, and
$\alpha=8$ gives it all back. At $k=3$ every order leaks **more** at equal utility. Phi sits with
KL3M-520M, and Pleias-1.2B remains the outlier.

Three pairs at the only budget where the constraint binds:

```
pair            log p_risky/token   anchor/token   alpha=2 advantage (nats/window)
Pleias-1.2B               -0.0012        -3.1187                            10.47
KL3M-520M                 -0.0027        -2.4236                             6.77
Phi-3.5-mini              -0.0254        -2.7140                             4.89
```

The advantage is **monotone in the memoriser's own log-probability of the protected tokens**: the
sharper the memoriser, the more a higher order buys. That is mechanically plausible --- a higher
order charges something closer to the worst step rather than the mean, and a sharp memoriser is
exactly the case where one step carries the passage --- but it is three points, where a monotone
ordering arises by chance one time in three, and neither the anchor's rate nor the gap between the
two orders the pairs the same way. It is a hypothesis and it is written down as one.

**Committed prediction, before Comma-7B runs.** Comma-7B's $\alpha=2$ advantage at $k=1$ will fall
in the position its per-token memoriser log-probability gives it among the four, i.e. Spearman
$\rho = 1$ over four pairs. A log-linear form is *not* predicted and would be wrong: the two slopes
implied by the three points are $-4.6$ and $-0.8$ nats per log unit, so only the ordering is
claimed. If the rank is wrong, the ordering is coincidence and the paragraph is deleted rather than
re-fitted; if it is right, four pairs with $\rho = 1$ is $p = 1/24$ under a random ordering and is
reported at exactly that strength, no more.

## Precision control: bfloat16 against float32 on one pair

Every checkpoint in this repository is *stored* in bfloat16 (`config.json`, all eight models), so
loading in float32 upcasts and buys accumulation precision, not weight precision. It also doubles
the memory, which is what stopped the Comma-7B pair sharing a card. Before running that pair in
bfloat16, the same grid was re-run on KL3M-520M in bfloat16 and compared cell by cell:

```
published k  alpha   float32   bfloat16   difference (nats per 50-token window)
        1.0      2      6.77       7.14        +0.37
        1.0      4      6.15       6.93        +0.78
        1.0      8      0.15       0.44        +0.29
        3.0      2     -0.06       0.32        +0.38
        3.0      4     -2.68      -2.62        +0.06
        3.0      8     -4.62      -4.91        -0.29
```

The bracket moves by $0.03\%$ ($-26.1$ to $-26.2$ nats on the protected tokens, $-23{,}024.6$ to
$-23{,}030.5$ on the anchor). The largest cell difference is $0.78$ nats per window, a factor of
$2.2$ in a quantity quoted in decades, so **a bfloat16 run supports a claim about the order of
magnitude and not about a factor of two**. It does not support a claim about the *sign* of a cell
near zero: at $k=3$, $\alpha=2$ the float32 run reads $-0.06$ and the bfloat16 run $+0.32$, which is
the same "no effect" read twice, and would be misreported as a direction. Comma-7B is run in
bfloat16 and its cells near zero are reported as zero.

## Also recorded: what a "window factor" is and is not

`nats_per_window` is the mean per-token log-probability difference multiplied by $50$, so
$e^{\text{nats}}$ is the **geometric mean** over 50-token windows of the factor by which an exact
window becomes less likely --- not the factor for any particular window, and not an arithmetic
average over windows. It scales exponentially with the window length, so the choice of $50$ is tied
to Table 1's metric and quoting it at another length changes the exponent proportionally. Both
qualifications belong in any sentence that carries this number.

## Scored on four pairs: the committed prediction fails, and nothing else predicts it either

`results/order_frontier_comma{,_matched}.csv` (bfloat16, bracket $-55.1$ / $-17{,}635.5$ nats, 0 of
48 cells outside it). At $k=1$, Comma-7B's $\alpha=2$ advantage is $8.36$ nats per window
($4.3\times10^{3}$) --- **second largest of the four, where its memoriser strength predicted third**.

    Spearman(memoriser strength, alpha = 2 advantage) = +0.800, exact two-sided p = 0.33
    predicted +1.000

So the ordering seen on three pairs was coincidence, and per the commitment above the paragraph is
deleted rather than re-fitted. The rank is not an artefact of Comma being the one bfloat16 run: the
control measured a bfloat16 bias of at most $+0.78$ nats per window and Comma leads KL3M-520M by
$1.59$.

Every other quantity available to a deployer was tested at the same time, on all four pairs and all
three orders, and reported whether or not it worked:

```
predictor                       alpha=2           alpha=4           alpha=8
memoriser log p / token   +0.80 (p=0.33)    +0.80 (p=0.33)    +0.80 (p=0.33)
anchor rate s(x)          -0.20 (p=0.92)    -0.20 (p=0.92)    -0.20 (p=0.92)
s(x) - memoriser rate     -0.20 (p=0.92)    -0.20 (p=0.92)    -0.20 (p=0.92)
F, distance from ceiling  -0.40 (p=0.75)    -0.40 (p=0.75)    -0.40 (p=0.75)
anchor parameter count    +0.00 (p=1.00)    +0.00 (p=1.00)    +0.00 (p=1.00)
protected tokens scored   -0.40 (p=0.75)    -0.40 (p=0.75)    -0.40 (p=0.75)
```

**What is left is a stable, unexplained pair effect.** The four pairs rank in the same order at
every order --- Pleias-1.2B, Comma-7B, KL3M-520M, Phi-3.5-mini --- so it is a property of the pair
and it reproduces across three decoders. Nothing measured explains it, and the spread it produces
is the result:

```
pair            k=1, alpha=4 matched-utility advantage
Pleias-1.2B                        1.7e+07 x safer
Comma-7B                           1.9e+05 x safer
KL3M-520M                              467 x safer
Phi-3.5-mini                            67 x safer
                 alpha=8:      3.3e+07 x safer  down to  1.7x MORE dangerous (Phi-3.5-mini)
```

$5.4$ decades at $\alpha=4$, and a sign change at $\alpha=8$. At the budget the mechanism's authors
publish, $k=3$, two of the four pairs leak **more** at equal utility at every order.

This is Section 6's title extended: the published budget determines neither the protection nor the
price, and neither does the order, the intervention rate, the anchor's surprisal rate, the
memoriser's strength, or how far the decoder is from saturation. A deployer choosing $\alpha$ is
choosing between $10^7\times$ safer and $1.7\times$ more dangerous with nothing to go on.

**Correction, same day.** The first four-pair run of `analysis/order_law.py` counted the bfloat16
precision control as a fifth series, so it compared KL3M-520M against itself. Excluding it moves the
spread at matched `F` from $2.2$--$7.7$ decades to $2.1$--$7.7$ and the median from $3.62$ to
$3.47$; the refutation is unchanged and every published figure now comes from the corrected run.
`order_law.py` skips `_bf16` files by default.
