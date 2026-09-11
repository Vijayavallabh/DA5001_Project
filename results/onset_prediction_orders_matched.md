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

## The negative is underpowered, and that is fixable. Committed before the seven-pair run.

"Nothing measured predicts it" currently rests on four pairs, where the best candidate scores
Spearman $+0.80$ at an exact two-sided $p = 0.33$. At that power a real predictor with
$\rho = 0.8$ and a coincidence are the same observation, so the claim as it stands is *not evidence
of absence*. A referee should say so, and the fix is arithmetic: the onset analysis already has
**seven** pairs, and all seven have memorisers on `attack_train` + `val`. At seven, $\rho = 1$ is
exact two-sided $p = 2/5040$ and even $\rho = 0.86$ is $p = 0.024$.

The three missing pairs are TinyComma-1.8B with a memorised Llama-3.1-8B (the only pair whose anchor
and risky model are different models), KL3M-1.7B, and Pleias-350M.

**One protocol change, stated before the runs.** Comma-7B needed `--dtype bfloat16` to fit, so the
present four-pair set is mixed precision, and the control measured a bfloat16 bias of up to $0.78$
nats per window --- enough to shuffle two pairs that sit close together, which is exactly what a rank
test is sensitive to. So **all seven pairs are run in bfloat16** and the rank test is computed on
that homogeneous set. The four-pair table in the paper is re-quoted from the bfloat16 runs for the
same reason; the float32 runs stay committed and the control quantifies the difference.

| outcome | reading |
|---|---|
| the best of the six candidates reaches $\lvert\rho\rvert \ge 0.86$ ($p \le 0.024$) over seven pairs | there is a predictor, it is named, and the claim becomes "the order's value is predicted by X and not by the published budget" |
| every candidate stays below $\lvert\rho\rvert = 0.7$ ($p > 0.1$) | the negative is earned at a power that can support it, and is reported as such with all six correlations shown |
| candidates land between, $0.7 \le \lvert\rho\rvert < 0.86$ | reported as inconclusive at seven pairs, with the number of pairs that would settle it |

Six candidates, fixed now: the memoriser's per-token log-probability of the protected tokens; the
anchor's surprisal rate $s(x)$; their difference; the fraction of the fidelity ceiling the audited
decoder captures at $k=1$; the anchor's parameter count; and the number of protected tokens scored.
No candidate is added after seeing the seven-pair result, and if one is, it is labelled post hoc.

**The precision control was itself underpowered.** It was run on one pair, KL3M-520M, and gave
$\le +0.78$ nats per window. With three pairs now run in both precisions the true range is
$-2.15$ to $+0.78$, and it is pair-dependent: under $0.8$ on KL3M-520M and Pleias-1.2B, and
$2.15$ (a factor of $8.6$) on Phi-3.5-mini. No rank changes in the control, but the manuscript's
"at most $+0.78$" was a one-pair extrapolation and has been corrected. The consequence for the plan
above: the three remaining pairs are run in **float32 as well**, so the seven-pair table and the
rank test are homogeneous float32, and the bfloat16 set becomes a seven-pair precision control
rather than the primary measurement.

## Seven pairs, bfloat16: two of the three cells are inconclusive and one negative is earned

`results/order_predictors{,_summary}.csv`, all seven onset pairs in bfloat16, advantage at the
published $k=1$ in nats per 50-token window:

```
pair              log p_r/tok   s(x)/tok       F   anchor params   a=2     a=4      a=8
Pleias-350M          -0.00646    -3.4954   0.782     353,424,384  10.82   17.05    16.47
Pleias-1.2B          -0.00124    -3.1195   0.767   1,195,468,800  10.50   16.39    16.76
Comma-7B             -0.00726    -2.3238   0.944   7,002,656,768   8.36   12.14     9.10
TinyComma-1.8B       -0.02858    -3.1474   0.852   1,758,562,304   8.00    1.66   -14.04
KL3M-520M            -0.00276    -2.4243   0.839     520,193,024   7.14    6.93     0.44
KL3M-1.7B            -0.00832    -2.2522   0.730   1,745,686,528   6.46   11.22    12.99
Phi-3.5-mini         -0.02550    -2.7140   0.840   3,821,079,552   3.86    2.05    -2.55

Spearman, exact two-sided p over all 7! orderings
candidate                     alpha=2          alpha=4          alpha=8
memoriser log p / token  +0.54 (0.236)    +0.71 (0.088)    +0.75 (0.066)
anchor rate s(x)         -0.57 (0.200)    -0.14 (0.783)    -0.04 (0.963)
s(x) - memoriser rate    -0.57 (0.200)    -0.14 (0.783)    -0.04 (0.963)
fraction of ceiling      -0.04 (0.963)    -0.43 (0.354)    -0.68 (0.110)
anchor parameter count   -0.43 (0.354)    -0.46 (0.302)    -0.50 (0.267)
protected tokens scored  -0.46 (0.302)     0.00 (1.000)     0.00 (1.000)
```

**Scored against the committed bands.** At $\alpha = 2$ the best candidate is $+0.54$, below $0.7$:
**the negative is earned at a power that supports it.** At $\alpha = 4$ and $\alpha = 8$ the best is
$+0.71$ and $+0.75$, inside the band the pre-registration called **inconclusive**, and that is what
is reported. The pre-registration also asked how many pairs would settle it: at $\rho = 0.75$ the
exact two-sided $p$ is $0.066$ at seven pairs, $0.037$ at eight, $0.026$ at nine, and clears the
committed $0.024$ at ten. So the honest statement is that the memoriser's own confidence on the
protected text is a *candidate* predictor at high orders which seven pairs cannot confirm or reject,
and that nothing else comes close at any order. Two caveats a referee should have: the seven pairs
are not seven independent draws (two Pleias, two KL3M), and the anchor's surprisal rate --- the
quantity this paper's own units result is built on --- is the *worst* performing candidate at
$\alpha=8$, at $-0.04$.

**The finding that does not need a predictor.** TinyComma-1.8B with a memorised Llama-3.1-8B is the
only pair whose anchor and risky model are different models, which is the mechanism's own
configuration. At $\alpha = 8$ and matched utility it is $-14.04$ nats per window: the protected
tokens are $1.2\times10^{6}$ times **more** likely than under the audited KL decoder. This is not an
interpolation artefact --- the matched budget $k = 5.25$ sits inside the grid, bracketed by
$k=4.0$ and $k=5.5$, and $\alpha=1$ at $k=1$ gives $-0.747$ nats per token against $\alpha=8$'s
$-0.50$ there. The mechanism is visible in the grid: a higher order charges something closer to the
worst step, so buying the same *average* fidelity costs it a much larger budget --- $5.25$ against
$1.0$ --- and that budget is spent tilting toward the memoriser on the steps that carry the passage.

Float32 runs of the three new pairs are in flight; nothing above reaches the manuscript until the
seven-pair table is homogeneous in float32, with bfloat16 kept as a seven-pair precision control.

## Ten pairs: deciding the cell the seven-pair run left inconclusive

Seven pairs put the best candidate at $\rho = 0.71$ and $0.75$ at $\alpha = 4$ and $8$, inside the
band this file called **inconclusive**, and named the fix: $\rho = 0.75$ reaches the committed
$p \le 0.024$ at **ten** pairs (exact two-sided: $0.066$ at seven, $0.037$ at eight, $0.026$ at
nine). Leaving it inconclusive when the fix is three LoRA fine-tunes of models under $4$B would be a
choice not to know.

Three anchors from the safe-model set have no memoriser yet and all three are already cached:
`alea-institute/kl3m-002-170m`, `alea-institute/kl3m-003-3.7b`, `PleIAs/Pleias-3b-Preview`. Each is
fine-tuned on `attack_train` + `val` exactly as the others were, with identical settings across the
three (`--target-modules all-linear --no-chat --epochs 40 --lr 3e-4 --rank 128 --batch 2 --accum 4
--max-len 0 --stop-loss 0.02`), then swept on the same 12-point `k` grid in `bfloat16`, which is the
precision the seven-pair set is homogeneous in.

**Entry gate, committed now.** A new pair enters only if (i) the bracket holds at every one of its
$48$ cells --- the served distribution's log-probability of the protected tokens strictly between
the anchor's and the memoriser's --- and (ii) the memoriser is materially better than its own anchor
on those tokens, at least a factor of $e$ per token. A model that fails either is **not a memoriser
on this corpus** and is excluded with its numbers reported here, not silently dropped.

**Two p-values, both committed.** Ten pairs is $10!$ orderings, too many to enumerate, so the naive
test is a fixed-seed Monte Carlo over $2\times10^{6}$ random permutations, which resolves $0.024$
to three decimals. The ten pairs are also **not ten independent draws** --- they would be four
KL3M, three Pleias, and one each of Phi, Comma and TinyComma --- so a family-clustered test is
reported beside it: rank the five *families* by their mean advantage and by their mean value of the
candidate, and compute the exact two-sided $p$ over all $5!$ orderings. That test is conservative
and its smallest attainable $p$ is $2/120 = 0.017$, so it can still decide.

| outcome | reading |
|---|---|
| a candidate reaches $\lvert\rho\rvert \ge 0.86$ over ten pairs **and** the family test agrees in sign at $\lvert\rho\rvert \ge 0.9$ | there is a predictor, it is named, and the paper says what a deployer should compute |
| the best candidate stays below $\lvert\rho\rvert = 0.7$ | the negative is earned at ten pairs and reported as the final answer |
| the two tests disagree, or the naive test lands between $0.7$ and $0.86$ | reported as still undecided, with both numbers and the family structure stated as the reason --- not resolved by choosing the friendlier test |

Memoriser strength differs across the ten pairs partly because the training settings differ between
the earlier runs; that is not a flaw here, because memoriser strength is one of the candidates being
*tested* rather than a variable being controlled. It does mean a positive result would need a
follow-up at matched training settings before it could be quoted as causal, and that is recorded now
so it cannot be skipped later.

**A limitation of this design, noticed while writing the code and recorded before the data lands.**
The three new anchors are two more KL3M and one more Pleias, so ten pairs still span **five**
families. The naive test gains power; the family-clustered test does not, and stays at $n = 5$
whatever is added, because the cached safe-model set has ten models in five families and the only
other un-memorised anchor (`common-pile/comma-v0.1-1t`) is a second Comma. So the conservative test
can only ever decide by reaching $\lvert\rho\rvert = 1$ ($p = 0.017$); at $\lvert\rho\rvert = 0.9$
it is $p = 0.083$ and stays there. If the two tests end up on opposite sides of the committed
threshold, the honest report is that ten pairs in five families cannot settle it, and the reason is
the family structure, not the number of pairs.

For the record, on the seven pairs already in hand the family view is *stronger* than the pair view
for the leading candidate --- the memoriser's own log-probability scores $\rho = +0.90$ on five
family means at $\alpha = 4$ and $8$ against $+0.71$ and $+0.75$ on seven pairs --- which is what
within-family noise diluting a real between-family effect would look like, and also what three
extra points on a five-point rank test would look like by chance. That is the ambiguity the
ten-pair run is meant to reduce and, on the family axis, cannot.

**The float32 control is now six pairs, and it makes the point better than five did.** With
TinyComma-1.8B added, the bfloat16 bias ranges $-2.34$ to $+0.78$ nats per window (a factor of
$10.4$) and is under $0.6$ on four of the six. It changes no pair's rank: the six-pair ordering at
$\alpha=2$ is identical under both precisions. And the *leading candidate* changes again --- on six
float32 pairs it is the anchor's surprisal rate at $-0.77$ for $\alpha=2$ and the fraction of the
ceiling at $-0.83$ for $\alpha=4$ and $8$, where on seven bfloat16 pairs it was the memoriser's own
log-probability. Three different sets, three different leaders, none reaching the committed
threshold: that is what a leading candidate looks like when it is noise, and it is the strongest
form of the negative available before the ten-pair run scores.

## An exploratory re-analysis, labelled as such: do the curves cross?

**Not pre-registered.** This came out of writing up the seven-pair result and re-reading the
matched-utility design. It re-analyses grids that are already committed, but the rule it applies ---
the noise floor --- was chosen *after* seeing that a naive sign test flags crossings of $0.4$ nats
per window, well inside the measured precision spread. It is reported as exploratory and nothing in
the pre-registered chain above depends on it.

The matched-utility comparison interpolates a budget. A simpler question needs no budget at all.
Each order traces a curve in the plane the mechanism trades in --- fidelity bought on the $x$ axis,
$L = \sum_t \log p_\theta(x_t)$ on the $y$ --- and if the four orders traced *one* frontier, matching
$x$ would match $y$. `analysis/order_crossings.py` sweeps $L(\alpha) - L(1)$ across the fidelity
range every order covers, counting a sign only when it clears that pair's own
bfloat16-against-float32 spread (or, for a pair with no float32 twin, the largest such spread over
the pairs that have one, which is the conservative choice).

```
7 of 21 (pair, order) cells cross, 13 are uniformly safer, 1 is uniformly more dangerous
```

**This is more nuanced than the matched-budget table suggested, and better for being so.** On most
cells a higher order really does help across the whole operating range. But on a third of them the
ranking *flips inside the range*, so which decoder is safer depends on an operating point the
published budget does not reveal --- and on TinyComma-1.8B with a memorised Llama-3.1-8B, the
mechanism's own configuration, $\alpha = 8$ is more dangerous at **100\%** of operating points, by
$2.6$ to $14.7$ nats per window.

**One thing this exposes about the appendix's own rule.** Comma-7B has no float32 twin, so it
borrows a floor of $2.34$ nats per window --- a factor of $10.4$ --- and under that floor its
$k = 3$, $\alpha = 8$ cell (a factor of $0.26$) is *inside the noise*, although the appendix's
blanket "within a factor of two of $1$ is no effect" would have read it as a direction. The blanket
rule is too lenient for the one pair whose precision is uncontrolled. Either Comma-7B gets a
float32 twin --- it needs $56$ GB for two 7B models and has not had a card --- or its cells are
quoted only as orders of magnitude. The appendix now says the latter explicitly.

## Is the order's value a property of the pair, or of the evaluation's seed?

Every number above is at `--seed-tokens 20`, the seed the attack uses, and Section 4 of the paper
shows that seed length is *not* innocuous: it moves the onset ratio enough to split the seven pairs
into two groups with no overlap. If it also moves the matched-utility advantage, then "a stable
property of the pair" is wrong and the finding belongs to one evaluation choice.

This is cheap to test and is committed before it runs: the same 12-point grid on **one** pair at
`--seed-tokens 10` and `80` against its committed `20`, in bfloat16, on KL3M-520M (the pair with the
shortest seed in words, so the three seeds span the widest range of context) and on Pleias-1.2B (the
longest, and the pair with the largest advantage).

| outcome | reading |
|---|---|
| the $k=1$ advantage moves by less than that pair's precision floor across the three seeds | the advantage is a property of the pair and the seed is not carrying it |
| it moves by more than a decade | the finding is seed-dependent, is quoted at one seed only, and the seed is added to the list of things a published $k$ does not reveal |
| it moves monotonically with the seed but by less than a decade | reported as a second-order sensitivity with the range, and the pair ranking is checked for stability rather than the levels |

The pair ranking is what the predictor tests use, so the ranking's stability under the seed is the
quantity that matters most; the levels are secondary. Nothing here is re-run at other seeds if the
first pair shows no movement, because that would be spending compute to confirm a null.

## The instrument checked against a decoded measurement

The first question a referee should ask about $L$ is whether it tracks what an adversary actually
recovers, and there is exactly one pair where both exist: Table 1's attack columns are the
TinyComma-1.8B anchor with the memorised Llama-3.1-8B, at the published $k=3$, which is also a pair
in the frontier set. At that budget:

```
alpha        1        2        4        8
oracle recall (Table 1)   0.097    0.054    0.004    0.001
L per token (this work)  -0.2412  -0.6127  -0.8886  -1.0456
```

Both are strictly monotone in $\alpha$ and order the four arms identically, so on the one pair where
a decoded measurement exists the rare-event functional agrees with it, at the same published budget,
without sampling or a judge. Four arms is Spearman $\rho = 1$ at exact two-sided $p = 2/24 = 0.083$,
which is weak, and it is the only decoded ground truth available; it is quoted at that strength.

The magnitudes are *not* comparable and should never be quoted as if they were. The recall ratio
from $\alpha=1$ to $\alpha=8$ is $97\times$; the exact-window factor implied by $L$ is
$e^{40} \approx 3\times10^{17}$. Recall is near-verbatim over a $50$-token window with a similarity
threshold and retries; $L$ is exact reproduction of every token. The functional is far the more
sensitive of the two, which is the point of using it, and also the reason its absolute value is
never quoted as a probability of anything an adversary would observe.

**Note on the Pleias-3B fine-tune, recorded while it runs.** Its token loss reached $0.0386$ at
epoch 15 and then *rose* ($0.0394$, $0.0437$), so it will not reach the `--stop-loss 0.02` the three
new pairs were committed to and will run all 40 epochs. The existing Pleias-1.2B memoriser plateaued
at $0.0295$, so this looks like a family property rather than a bug: Pleias models do not drive this
corpus below about $0.03$ at these settings. The run is **not** being restarted with different
settings after seeing its loss curve. It finishes as committed, and if the merged model fails the
entry gate above --- the memoriser materially better than its own anchor on the protected tokens ---
the pair is excluded and that exclusion is reported here, which is what the gate was written for.

**The Pleias-3B fine-tune diverged and was stopped.** After the plateau it went $0.0386 \to 0.0437
\to 0.0786 \to 0.1291$ over epochs 15-20, so it was killed at epoch 20 rather than run to 40 to
produce a knowingly-diverged model. This is an operational decision about a training run, not an
analysis decision about a result, and it is recorded with its reason: **divergence of the loss, not
the sign or size of any advantage, which had not been computed.** One retry is committed now, before
it runs, at `--lr 1e-4 --stop-loss 0.03` --- a lower rate because the run diverged, and the looser
floor because the existing Pleias-1.2B memoriser plateaued at $0.0295$ and the family evidently does
not drive this corpus below about $0.03$. If the retry also fails the entry gate, Pleias-3B is
excluded and the ten-pair set becomes nine, with the exclusion reported here.

**Eight pairs exist and are deliberately not being scored.** KL3M-170M landed while KL3M-3.7B and
the Pleias-3B retry were still training, and the eight-pair numbers are already computable --- the
best candidate is $-0.69$ at $\alpha = 8$, which would fall inside the "earned negative" band. That
is **not** the reported result. The pre-registration committed to ten pairs (nine if one fails the
entry gate), and scoring at eight because the answer looks settled there, then scoring again at ten,
is two looks at the same data. The number that gets reported is the one at the committed endpoint,
whichever direction the last two pairs move it, and the eight-pair figure is written down here only
so that it cannot later be presented as if it had never been seen.

## Scored: the seed moves the levels a little and the ranking not at all

`results/order_seed.csv`, from `analysis/order_seed.py`. Both pairs at `--seed-tokens 10` and `80`
against their committed `20`, on the same 12-point grid in bfloat16, compared against the same
noise floor the crossing test uses --- that pair's own bfloat16-against-float32 spread.

```
9 of 24 cells move beyond their pair's precision floor, 1 by more than a decade, 0 change sign
```

**The middle band fires, and the part that matters is the cleanest.** At the published $k=1$, where
the constraint actually binds and every headline number lives, the twelve cells move by
$-0.59$ to $+1.42$ nats per window and only three clear their floor; the largest single move is
KL3M-520M at $\alpha=8$, from $0.44$ to $1.86$ nats, which is "no effect" read twice. At $k=3$
Pleias-1.2B moves more (up to $-2.41$, a little over a decade) --- that is the saturated region
where the audited decoder already has $97\%$ of the ceiling and every order is compressed against
it, so it is the region the paper already says the certificate has nothing to say about.

**No cell changes sign, and the pair ranking is stable at every order and every seed:**
Pleias-1.2B leads KL3M-520M by $3.4$, $9.5$ and $16.3$ nats at seed 20, by $3.1$, $8.2$ and $14.4$
at seed 10, and by $3.2$, $8.9$ and $15.1$ at seed 80. The ranking is what every predictor test
consumes, so its stability is the quantity that mattered, and it holds across a factor of eight in
seed length. The advantage is a property of the pair.

Per the pre-registration, nothing is re-run at other seeds: the first pair showed no movement at the
budget that matters, and spending compute to confirm a null is what the sentence was written to
prevent.

## Pleias-3B is excluded, as the entry gate provided for

The one committed retry at `--lr 1e-4 --stop-loss 0.03` reached a minimum of $0.0326$ at epoch 21
and then turned, exactly as the first run did:

```
ep 18  0.0343    ep 21  0.0326    ep 24  0.0384    ep 27  0.0670
ep 19  0.0332    ep 22  0.0342    ep 25  0.0424
ep 20  0.0327    ep 23  0.0350    ep 26  0.0523
```

It was stopped at epoch 27 rather than run to 40 to produce a model worse than its own minimum. The
commitment was **one** retry; a second would be tuning until it worked, so **Pleias-3B is excluded
and the set is nine pairs**, which is what the entry gate was written to allow.

The exclusion is itself a small finding and is reported rather than buried: **Pleias-350M and
Pleias-1.2B memorise these 608 excerpts under LoRA rank 128 and Pleias-3B does not**, at either
$3\times10^{-4}$ or $10^{-4}$, plateauing near $0.033$ and then diverging both times. No claim is
made about why. It does mean the nine pairs are four KL3M, two Pleias, and one each of Phi, Comma
and TinyComma --- five families still, so the family-clustered test is unchanged at $n = 5$, as this
file predicted before any of it ran.

At $\rho = 0.75$ the exact two-sided $p$ is $0.026$ at nine pairs against the committed $0.024$
threshold, so a candidate at that strength would land just outside it and be reported as
inconclusive by a hair. That is a worse position than ten pairs would have given and it is stated
plainly rather than softened: the set is what the models allow, not what the test would prefer.

## Scored at nine pairs: the negative is earned

`results/order_predictors{,_summary}.csv`. KL3M-3.7B's memoriser reached loss $0.0187$ at epoch 9
with sampled recall $0.854$, and its bracket holds at all 48 cells, so it enters; Pleias-3B is
excluded as recorded above. Nine pairs, matched-utility advantage at the published $k=1$, nats per
50-token window:

```
Pleias-350M 10.82   Pleias-1.2B 10.50   Comma-7B 8.36   KL3M-170M 8.09
TinyComma-1.8B 8.00   KL3M-3.7B 7.22   KL3M-520M 7.14   KL3M-1.7B 6.46   Phi-3.5-mini 3.86

Spearman, exact two-sided p over all 9! orderings
candidate                     alpha=2          alpha=4          alpha=8
memoriser log p / token  +0.35 (0.359)    +0.42 (0.270)    +0.48 (0.194)
anchor rate s(x)         -0.48 (0.194)    -0.10 (0.810)    +0.02 (0.982)
s(x) - memoriser rate    -0.48 (0.194)    -0.10 (0.810)    +0.02 (0.982)
fraction of ceiling      -0.08 (0.843)    -0.45 (0.230)    -0.53 (0.148)
anchor parameter count   -0.35 (0.359)    -0.30 (0.437)    -0.23 (0.552)
protected tokens scored  -0.48 (0.194)    -0.12 (0.776)    -0.07 (0.880)
```

**The best candidate over nine pairs is $\lvert\rho\rvert = 0.53$, below the committed $0.7$: the
negative is earned at every order and that is the final answer.** The leading candidate at seven
pairs --- the memoriser's own log-probability, at $+0.71$ and $+0.75$ --- has fallen to $+0.42$ and
$+0.48$. Adding two pairs halved it, which is what a coincidence does when it meets more data and is
the reason the seven-pair cell was reported as inconclusive rather than quoted.

**The conservative test cannot decide and says so.** On five family means the memoriser's rate is
$+0.90$ at $\alpha = 4$ and $8$, exact $p = 0.083$ over all $5!$ orderings --- one adjacent swap from
perfect, and not significant. It has read $+0.90$ at five, seven, eight and nine pairs, because
family means barely move when a family gains a member, and it cannot be pushed below $p = 0.017$
without a sixth family that the cached model set does not contain. So it is the one signal that
persists, it does not reach the threshold, and it does not overturn the naive test. Both numbers are
reported; neither is chosen over the other.

**What this licenses the paper to say.** Nothing a deployer can compute --- the memoriser's own
confidence, the anchor's surprisal rate, their difference, the anchor's size, the number of tokens,
or how far the audited decoder is from its own fidelity ceiling --- predicts what a higher Renyi
order is worth at matched utility, across nine pairs where that worth spans from $10^{7}$ times
safer to $10^{6}$ times more dangerous. The one candidate that survives at the family level is the
memoriser's own confidence on the protected text, at a strength five families cannot resolve, and
the paper says exactly that.

## Seven families: deciding the one signal five families could not resolve

Nine pairs earned the negative on the naive test (largest $\lvert\rho\rvert = 0.53$), but the
conservative family-clustered test left one candidate standing --- the memoriser's own
log-probability per token, at $\rho = +0.90$, exact $p = 0.083$ over all $5!$ orderings. That is one
adjacent swap from perfect on five points, and this file recorded before the nine-pair run that it
**could not be improved** by adding more KL3M or Pleias anchors, because the cached safe-model set
is ten models in five families. It also recorded what would settle it: a sixth family.

Three are available and already cached, so no download is needed and nothing gated is fetched:

* `meta-llama/Llama-3.2-1B` and `meta-llama/Llama-3.2-3B-Instruct` --- a **Llama** family whose
  anchor is a Llama model. (The existing TinyComma pair's *risky* model is a memorised Llama-3.1-8B,
  but its anchor is TinyComma, and every candidate here is a property of the anchor or of the pair,
  so these are new. A self-paired Llama-3.1-8B was considered and **rejected**: it would share its
  memoriser exactly with the TinyComma pair, which is a tighter dependence than sharing a family.)
* `Qwen/Qwen2.5-7B-Instruct` --- a **Qwen** family. It is the judge used in Section 5, which is
  noted for transparency; the frontier analysis uses no judge, so there is no path between the two.

Each is fine-tuned on `attack_train` + `val` with the settings already used for the previous three
(`--target-modules all-linear --no-chat --epochs 40 --lr 3e-4 --rank 128 --batch 2 --accum 4
--max-len 0 --stop-loss 0.02`), then swept on the same 12-point grid in `bfloat16`. The **entry
gate is unchanged**: the bracket must hold at all 48 cells and the memoriser must beat its own
anchor on the protected tokens by at least a factor of $e$ per token. Two of the three anchors are
instruction-tuned, as Phi-3.5-mini already is; that is noted, not controlled.

That would give **twelve pairs in seven families**. At seven families $\rho = 1$ is exact
$p = 2/5040$ and one adjacent swap is $p \approx 0.024$, so the family test can finally decide
rather than report the same $+0.90$ it has reported at five, seven, eight and nine pairs.

| outcome | reading |
|---|---|
| the family test reaches $\lvert\rho\rvert \ge 0.86$ at $p \le 0.024$ on seven families | the memoriser's own confidence on the protected text **is** the predictor, it is named, and the paper tells a deployer to compute it. The naive test's disagreement is then explained by within-family noise and both are reported |
| it falls below $\lvert\rho\rvert = 0.7$ | the $+0.90$ was an artefact of five points, the negative is earned on both tests, and the paper says so without hedging |
| it stays between | seven families still cannot resolve it, and that is the final answer: the paper reports both tests, the number of families that would be needed, and stops |

Twelve pairs is more than $9!$ can enumerate, so the naive test switches to the fixed-seed Monte
Carlo over $2\times10^{6}$ permutations that `order_predictors.py` already implements and labels.
Whatever the twelve-pair naive number is, it is reported --- including if it rises above the $0.53$
that earned the negative at nine.

## A second protected corpus, committed before anything is trained on it

Every extraction number in this paper comes from one corpus: sixteen English genre novels from
CopyBench. Limitations says so, and it is the largest single caveat --- the nine "independent"
robustness cells are nine re-analyses of the same works, and the order results inherit that.

`analysis/build_gutenberg_excerpts.py` builds a second corpus in the identical shape from the 50
public-domain books already cached for `anchor_scaling.py`: 600 excerpts, 925-character prefix and
225-character continuation, Gutenberg header and licence stripped, whitespace collapsed, taken at
evenly spaced offsets inside each book's body. Public-domain text is not "protected" in the legal
sense and that is not what is being tested; what is being tested is whether the geometry the paper
measures belongs to the pair or to those sixteen novels.

**Design: the anchor is held fixed and only the protected work changes.** Two anchors already in the
set --- KL3M-520M and Pleias-1.2B --- get a second memoriser each, trained on the Gutenberg
excerpts with the same settings, and are swept on the same 12-point grid. Everything but the
corpus is identical to the run already reported, including the ordinary-generation side, so the
comparison is within-anchor.

The reader is deliberately separate from `dap.shared.load_prompt_corpus` (`analysis/corpus_file.py`,
used by `--corpus-file` on both scripts): the committed prompt sets under `data/` are not to be
modified and adding a file to `SOURCE_FILES` would change what every other script sees. The
instruction prefix is identical, so the memoriser and the sweep see the same form.

| outcome | reading |
|---|---|
| both anchors' $k=1$ advantages land within their own precision floor of the CopyBench numbers | the geometry is a property of the pair and the single-corpus caveat, while still true of the onset results, does not reach the order results |
| the advantages move but keep the same sign and the KL3M-below-Pleias ordering | corpus-sensitive in level and not in rank, reported with both numbers, and the rank is what the predictor tests consume |
| a sign flips, or the ordering between the two anchors inverts | the order results are corpus-specific, must be quoted for CopyBench only, and the corpus joins the list of things a published $k$ does not reveal |

The entry gate is unchanged and matters more here than anywhere: these anchors were trained on
Common Pile and KL3M's legal corpora, which may already contain some of these public-domain books,
so an anchor that is *already* fluent on a passage leaves less for the memoriser to add. The bracket
--- the served distribution's log-probability of the protected tokens strictly between the anchor's
and the memoriser's --- is what detects that, and any pair failing it is excluded with its numbers
reported.

## Every pair now has its own precision floor, and the crossing count changes because of it

Three pairs were borrowing a floor: Comma-7B, whose two $7$B models had never had a card in
`float32`, and the two KL3M anchors added last. All three now have `float32` twins, so all nine
pairs carry a measured floor of their own:

```
kl3m 0.78   pleias 0.55   phi 2.15   pleias350 2.34   kl3m17b 0.11
tinycomma 0.52   comma 1.14   kl3m170m 0.47   kl3m37b 0.32   (nats per 50-token window)
```

Comma-7B's own floor is $1.14$, not the $2.34$ it was borrowing, and the borrowed value was the
largest measured anywhere --- so the conservative choice was, as intended, conservative. With the
real floors the exploratory crossing test moves from **$7$ of $27$ cells crossing** to **$12$ of
$27$**, with $14$ uniformly safer and $1$ uniformly more dangerous. The change is entirely the
floors: no grid was re-run and no rule was altered. It makes the finding stronger, which is exactly
why it is recorded here as a consequence of a measurement rather than presented as if the number had
always been $12$.

Across all nine pairs the `bfloat16`-against-`float32` difference is $-2.34$ to $+0.86$ nats per
window over 54 cells, with a **median absolute difference of $0.30$** --- so the large values are
two pairs (Phi-3.5-mini and Pleias-350M) and not the norm. The manuscript's "Comma-7B is the
exception twice over" sentence is withdrawn: it no longer is.

## The bracket's tolerance is too tight at the saturated end, and Llama-3.2-1B found it

Llama-3.2-1B's memoriser is admissible ($0.690$ sampled recall) and its bracket is enormous ---
$-319.3$ nats on the protected tokens against the anchor's $-18{,}159.8$ --- yet the run reports
**5 of 48 cells outside it**. All five are at $k = 10$ and $14$, the top of the grid, and all five
overshoot the *upper* bound by at most $1.3$ nats:

```
alpha 1 k 10   -318.009      alpha 2 k 14   -315.805      alpha 8 k 14   -318.972
alpha 1 k 14   -313.623      alpha 4 k 14   -317.939      (risky = -319.3)
```

At the top of the grid $\theta \to 1$ and the served distribution *is* the risky model, so $L$ must
approach the upper bound exactly; what crosses it is accumulated floating-point error in a sum of
$6{,}500$ `bfloat16` log-probabilities, which is order one nat. The check's tolerance is
$10^{-6}\lvert L_{\text{safe}}\rvert = 0.018$ nats, tighter than the arithmetic it is checking.

Every other pair passes only because its memoriser is far stronger --- $-8$ to $-55$ nats --- so the
constrained values never come within a nat of the ceiling. **The gate is failing on a property of my
tolerance, not of the pair.**

**How this is being resolved, stated before the numbers are used.** Loosening a gate after a pair
fails it is the classic goalpost move, so the amendment is made by a rule that does not depend on
the outcome and is applied uniformly to all pairs, with the effect on each reported:

* The bracket exists to catch an **inverted or mis-specified** pair --- the split bug, where the
  memoriser sat $14{,}063$ nats *below* the anchor. It was never meant to resolve one nat at
  saturation.
* The tolerance becomes `max(1e-6 * |L_safe|, 1e-4 * (L_risky - L_safe))`: a fixed fraction of the
  bracket's own width, which is the scale the accumulated error lives on. For Llama-3.2-1B that is
  $1.78$ nats; for KL3M-520M, $2.3$.
* No tolerance of this size can mask an inversion, which is a sign change of thousands of nats.
* The worst overshoot, in nats and as a fraction of the bracket width, is now printed for every pair
  so a reader can see how close any of them came.

If the amendment changes the verdict for any pair *other* than Llama-3.2-1B, that is reported here.

## The upper "bound" is not a bound, and a counter-example settles it

The amended tolerance changed **nothing** for any of the nine existing pairs (0 cells outside before
and after, worst excursions $0.01$ to $3.06$ nats against tolerances of $1.8$ to $2.5$) --- so it is
demonstrably not an outcome-driven loosening. But Llama-3.2-1B still fails, with a worst excursion
of $+5.71$ nats, $0.032\%$ of its bracket width. That is too large to be `bfloat16` accumulation,
and chasing it found a real error in the check.

**$L(\theta)$ is not monotone in $\theta$, so the risky model's own log-probability is not an upper
bound on it.** Two steps and three tokens are enough:

```
p_s = [[.10 .80 .10], [.80 .10 .10]]   p_r = [[.90 .05 .05], [.05 .90 .05]]   target = (0, 0)
theta   0.00     0.25     0.50     0.75     0.90     1.00
L      -2.526   -1.830   -1.692   -2.183   -2.700   -3.101
```

At $\theta = 0.5$ the served distribution gives the true tokens $e^{1.41}$ times *more* mass than
the risky model does at $\theta = 1$. Mixing the anchor in helps whenever the anchor is right where
the risky model is wrong, and summed over thousands of steps a partial tilt can beat a full one.
This is a property of the geodesic, not of any implementation.

It also explains exactly *which* pairs trip it: a weak memoriser leaves the anchor competitive on
many tokens, so the mixture wins more often. Llama-3.2-1B's memoriser is the weakest in the set at
$-319.3$ nats; every other pair sits between $-8$ and $-187$, where $p_r$ dominates and no mixture
helps. **The gate was failing on a pair that stressed a check I had stated too strongly.** An early
note in this file said the bracket was not a theorem; it was then used as one, which is the error.

**The corrected gate, and what it is allowed to decide.** The bracket's real job --- the one it did,
catching a memoriser sitting $14{,}063$ nats *below* its anchor because the probe was reading a
held-out novel --- is the lower side. That side is kept as the gate, together with the
pre-registered requirement that the memoriser beat its anchor by at least a factor of $e$ per token.
The upper side becomes a **reported diagnostic**: the worst excursion in nats and as a fraction of
the bracket width is printed for every run, so a reader sees how far any pair went.

Llama-3.2-1B passes the corrected gate: $-319.3$ against an anchor at $-18{,}159.8$, and no cell
below the anchor. **Because admitting a pair after amending a gate it failed is exactly the move a
reader should be suspicious of, the ten-pair result is reported both with and without it**, and the
amendment is justified by the counter-example above rather than by anything the pair measured.

## Scored: the geometry survives a change of protected corpus

`results/order_frontier_gut_{kl3m,pleias}_bf16{,_matched}.csv`. Same anchors, same settings, same
grid, same ordinary-generation side; only the protected work changed, from sixteen copyrighted
novels to 600 excerpts of 50 public-domain books. Both memorisers pass the corrected gate
(KL3M $-25.8$ nats against an anchor at $-21{,}655.8$; Pleias likewise) and no cell of either sits
below its anchor.

```
pair          k  alpha  CopyBench  Gutenberg   diff  floor  beyond?
KL3M-520M     1    2       7.14       7.24   +0.10   0.78   no
KL3M-520M     1    4       6.93       8.16   +1.23   0.78   YES
KL3M-520M     1    8       0.44       2.53   +2.09   0.78   YES
KL3M-520M     3    2/4/8   0.32/-2.62/-4.91  within 0.56    no
Pleias-1.2B   1    2      10.50      10.12   -0.38   0.55   no
Pleias-1.2B   1    4      16.39      16.51   +0.12   0.55   no
Pleias-1.2B   1    8      16.76      17.73   +0.97   0.55   YES
Pleias-1.2B   3    2/4/8   4.21/4.73/3.30    within 0.44    no
```

**The second band fires and the important half is the cleanest.** Nine of the twelve cells move less
than their pair's own precision floor; three exceed it, the largest by $2.09$ nats per window (a
factor of $8$, in a quantity quoted in decades). **No cell changes sign, and Pleias-1.2B leads
KL3M-520M at every order on both corpora** --- the ranking is what every predictor test consumes,
and it survives.

Two measurements worth recording beside it. The anchors are **not** markedly more fluent on the
public-domain books than on the copyrighted novels: $-2.42$ against $-2.28$ nats per token for
KL3M-520M and $-3.12$ against $-3.01$ for Pleias-1.2B, a $5\%$ difference, so the two corpora are
comparably hard for these anchors and the comparison is not confounded by exposure. And the two
Gutenberg memorisers differ in strength from their CopyBench twins in opposite directions
($-0.00271$ against $-0.00276$ for KL3M, $-0.0251$ against $-0.0012$ for Pleias), which is one more
reason the level moves while the rank does not.

**What the paper may now say.** The single-corpus caveat is true of the onset results, which rest
entirely on those sixteen novels, and it does **not** reach the order results: the matched-utility
geometry reproduces on a second, disjoint corpus with the anchor held fixed. Two anchors is not a
demonstration that it holds for all; it is a demonstration that the first corpus was not doing the
work.

**A third anchor on the second corpus, committed before it runs.** The corpus result rests on two
anchors, both of which sit in the upper half of the advantage range. Phi-3.5-mini is the pair with
the *smallest* advantage in the whole set ($3.86$ nats per window at $k=1$, $\alpha=2$, against
Pleias-350M's $10.82$) and belongs to a third family, so it is the useful third point: if the
geometry is corpus-invariant only where the advantage is large, that is where it would show. Same
design, same settings, same bands as above. A fourth anchor is not planned; two outcomes -- holds at
both ends, or holds only at the top -- are what three points can distinguish, and a fourth would not
change which.
