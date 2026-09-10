# Pre-registration: does the temperature elasticity survive matching on memoriser strength?

Committed before the analysis is run. Zero GPU: it re-reads two sweeps that are already committed.

## The confound this addresses

The decoder warps **both** logit vectors before the KL solve (He et al., App. B), so setting
tau = 0.4 sharpens the risky model as well as the anchor. The unconstrained memoriser is therefore
stronger in the warped arm than in its own control --- k = -1 near-verbatim recall 0.904 against
0.519 (KL3M-520M) and 0.964 against 0.909 (Pleias-1.2B) --- and a stronger memoriser reaches the
threshold at a lower budget. The measured onset in the warped arm is thus **too low**, and the
elasticity d log(onset) / d log s(x) --- +0.72 [+0.41, +0.84] and +0.61 [+0.40, +0.96] --- is
**biased toward 0**, which is the direction that argues for the constant-nats null we are refuting.

The two facts the elasticity is built from are not in doubt; what is in doubt is how much of the
onset's rise is the anchor's s(x) and how much is the memoriser being handed a sharper distribution.

## The intervention

Restrict both arms to the passages the memoriser reproduces in **both**, then recompute each arm's
onset and the elasticity on that subset alone. On it the two arms' unconstrained strength is matched
by construction, so what differs between them is the anchor's rate and nothing else.

**Threshold, fixed now: k = -1 single-query near-verbatim recall >= 0.7 in both arms.** Chosen for
power before any onset was computed: it leaves n = 38 (KL3M-520M) and n = 86 (Pleias-1.2B), where
0.9 would leave 29 and 85 and 0.5 would leave 45 and 87. Everything else --- the k grid, the
threshold of four LCS words, the interpolation inside the bracket, the bootstrap over passages ---
is unchanged, and the control arm is re-scored on the same subset so the comparison is like for
like.

## Bands

| outcome | reading |
|---|---|
| matched elasticity excludes 0 on both pairs | the within-pair refutation of constant nats survives with strength controlled; this is the decisive outcome |
| excludes 0 on one pair only | partial; report both, and rest the claim on the pair that survives plus the cross-pair analysis |
| includes 0 on both pairs | the within-pair evidence is confounded by memoriser strength; **withdraw it as decisive** and rest the units claim on the matched-context cross-pair analysis alone |

**A separate directional prediction.** Because the strength gap biased the unmatched elasticity
down, the matched estimate should be **at or above** the unmatched one: >= +0.72 (KL3M-520M) and
>= +0.61 (Pleias-1.2B), to within the bootstrap. A matched estimate materially *below* its unmatched
value contradicts the account of the bias given above and will be reported as such, not explained
away.

The subset is smaller than the full sweep, so its intervals will be wider; an interval that widens
to include 0 while the point estimate holds is reported as underpowered, not as a refutation.

---

## Result: the elasticity does not move at all

`analysis/matched_strength.py --out results` -> `results/matched_strength.csv`.

| arm | $n$ | $k=-1$ recall, control / warped | elasticity | 95% CI | unmatched |
|---|---|---|---|---|---|
| KL3M-520M tau 0.4 | 38 of 100 | 0.947 / 0.993 (was 0.519 / 0.904) | **+0.72** | $[+0.08, +1.00]$ | +0.72 |
| Pleias-1.2B tau 0.4 | 86 of 100 | 0.996 / 0.999 (was 0.909 / 0.964) | **+0.61** | $[+0.38, +1.03]$ | +0.61 |

Both intervals exclude 0, so the **decisive band is met**: the within-pair refutation of constant
nats survives with unconstrained memoriser strength matched. The directional prediction --- that
the matched estimate would come in at or above the unmatched one --- holds, at equality: to two
decimals the estimate does not move on either pair, although the pieces it is built from do
(KL3M's onsets go 2.492 -> 2.462 and 3.603 -> 3.550, Pleias's 2.780 -> 2.748 and 3.740 -> 3.694).
Dropping 62 of 100 passages from the KL3M pair and closing a strength gap of 0.519 against 0.904
changes the answer in the third decimal. Whatever the warp does to the onset, it does not do it
through the memoriser.

**One thing the matched analysis gives up.** On the full sample both intervals exclude 1 as well as
0, so the response is sub-proportional there; on the matched subsets, with 38 and 86 passages, the
upper ends reach 1.00 and 1.03. Sub-proportionality is a full-sample statement and is reported as
one. What survives matching is the part that refutes the null: the onset moves with $s(x)$, and by
more than nothing.
