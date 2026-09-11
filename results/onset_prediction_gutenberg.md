# Does the derived onset law hold on a corpus it has never seen?

**Committed 2026-09-11, before any decoding on the second corpus.** The predictions below were
computed from two teacher-forced forward passes per passage and no attack at all; they are in
`results/onset_theory_gutenberg.csv`, written and committed in the same commit as this file.

## Why this is the experiment worth running

The plan's own risk list says it plainly: *"Everything runs on 16 English genre novels; the nine
'independent' robustness cells are nine re-analyses of the same 16 works."* feat-080 answered that
for the **order** results -- the matched-utility geometry reproduces on a second, disjoint corpus.
It did not answer it for the **onset**, which is the paper's positive contribution and the one place
a law is claimed rather than a measurement reported:

> P1  onset / s(x) = 1 - s_r(x)/s_s(x)

Every test of P1 so far has been on those sixteen novels. P1 is parameter-free and needs no
decoding, so a second corpus gives something the appendix cannot: an **out-of-sample** test, where
the prediction is written down before the measurement exists.

## The three pairs and their predictions

The three anchors that already carry a Gutenberg memoriser from feat-080, so no new fine-tune is
needed and nothing about the memorisers was chosen after seeing an onset. 100 excerpts of 50
public-domain books, the same passages feat-080 used, `--limit 100` as every other onset pair.

```
pair                          s_s     s_r    pred onset   pred ratio   r(x) q10   r(x) q25
KL3M-520M (Gutenberg)       2.3665  0.1487     2.218        0.9372       1.868      2.023
Pleias-1.2B (Gutenberg)     2.8087  0.3453     2.463        0.8771       1.957      2.204
Phi-3.5-mini (Gutenberg)    2.8488  0.0362     2.813        0.9873       2.170      2.378
```

Their CopyBench twins -- same anchors, same architecture, same settings, different protected work --
are already measured, which is what makes this a comparison and not a single reading:

```
pair (CopyBench)     pred    measured   pred/meas
KL3M-520M           2.200      2.543       0.865
Pleias-1.2B         2.845      2.819       1.009
Phi-3.5-mini        2.829      2.628       1.076
```

So on the corpus the law was developed on, these three run **0.865 to 1.076**: within 14%.

## Committed bands

Scored on single-query near-verbatim recall at `--thresh 0.01`, the same definition and the same
threshold as every onset in the paper, with `analysis/onset.py`'s own bracket-and-interpolate rule.

| outcome | reading |
|---|---|
| all three pred/meas in **[0.85, 1.15]** | the law transfers out of sample, no worse than on the corpus it was derived on. P1 is a law, not a fit to sixteen novels |
| all three in **[0.7, 1.4]**, at least one outside [0.85, 1.15] | it transfers in order of magnitude and not in precision; report the range and stop quoting a two-decimal agreement |
| **any pair outside [0.7, 1.4]**, or no crossing on the grid | the near-determinism approximation is corpus-specific. The paper says so, in the onset section and not only in Limitations |

**Secondary, P2.** The population onset should sit at a *low quantile* of the per-work requirement
r(x), because the cheapest works leak first. Committed: the measured onset falls between `q10` and
the median of r(x) on all three pairs. On CopyBench it landed at `q25` to within 0.01 nats on the
two headline pairs; nothing that specific is committed here, because q25 to two decimals on two
pairs is exactly the kind of precision this file exists to stop over-reading.

**Entry gate, unchanged.** A pair enters only if its **sampled** `k = -1` recall is at least 0.10
(AGENTS.md caution (a); greedy recall lies). The `k = -1` and `k = 0` baselines run on the same
passages and seeds as every budget, as they must.

**The grid**, fixed now so it cannot be chosen to bracket an answer:
`-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2`. It brackets every prediction above with the
lowest point below every `q10` and the highest well above every median, so a crossing that lands at
either end is a **finding about the grid** and is reported as one -- see AGENTS.md caution (g), the
bootstrapped crossing that was narrow only because it was conditioned on the resamples that crossed.

**What will not happen.** No second grid, no re-threshold, no re-run at another seed to see if it
moves. The bands above are the whole of the scoring rule.

## One measurement made while building this, reported because it bears on the numerator

`budget_path.py` measures the anchor's `s_mean` on the **instruction-prefixed** context
(`"Complete the prefix:\n"` + prefix), while `onset_theory.py` measures the risky model's `s_r` on
the **raw** prefix -- so the published ratio `s_r/s_s` mixes two conditionings. Measured on
KL3M-520M over the same 100 CopyBench passages: the header moves `s_s` by a median of
**-0.0012 nats per token, -0.05%** (range -0.019 to +0.011). It is not a bias worth correcting and
the published ratios stand. The Gutenberg branch added here measures both sides on the raw prefix,
so it does not inherit the mismatch at all.

## Addendum, committed while the sweeps were still running and before any was scored

Re-reading `sections/onset.tex` against the file above: **the paper already rejects
Eq.~\eqref{eq:req} as a predictor of the onset**, and the framing above does not say so clearly
enough. What it rejects, precisely, is the *directional* claim. On the seven CopyBench pairs the
rank correlation between the predicted ratio $1 - s_r/s_s$ and the measured one is $-0.18$ where the
derivation requires it positive; the pair carrying by far the most residual surprisal has the
highest ratio of all. Three committed rules all missed, with held-out errors of $0.352$, $0.251$ and
$0.481$ nats.

What was *not* rejected is the **level**. On the three pairs used here the CopyBench pred/meas runs
$0.865$, $1.009$, $1.076$: the equation puts the onset in the right place to within about $14\%$
while ordering the pairs wrongly. That is the distinction this experiment is in a position to test
and the one it is hereby restricted to:

- **What is being asked.** Is the level agreement a property of the geometry, or of those sixteen
  novels? A second corpus is the only way to tell, and the prediction for it was written down before
  any decoding.
- **What a pass does NOT license.** It does not reinstate Eq.~\eqref{eq:req} as a predictor and it
  does not make it a law. The direction stays refuted; three pairs cannot revive it, since with
  $n = 3$ the smallest attainable exact $p$ is $1/3$. If the three happen to rank correctly that
  will be *stated with its $p$* and given no weight, exactly as the seven-pair inversion is.
- **What a failure means.** If the levels miss on the second corpus the way the directions missed on
  the first, then even the level agreement is a coincidence of one corpus, and the onset section
  should say that the equation is right about what a decoder must *afford* and carries no
  quantitative content about where leakage begins.

The numeric bands, the grid, the threshold and the entry gate above are unchanged. Only the
*reading* of each outcome is narrowed, and it is narrowed before the numbers exist.

## Unrelated, found by the same read-through: the grid-ceiling rule was not applied to the main table

`sections/appendix_seed.tex` commits, for the seed arms, to *"extending the grid whenever the
[bootstrap no-crossing] fraction rises materially above the others and to reporting both grids"*.
`results/onset_ci.csv` says the fraction over the eight onset rows is $0.0$--$0.1\%$ on five of
them, $4.3\%$ on KL3M-1.7B and $6.9\%$ on the Pleias-350M $n{=}100$ arm. The rule was written for
the seed arms and never applied to the main-text table, where KL3M-1.7B is one of the two pairs
whose onset ratio exceeds $1$ --- a headline claim.

Its grid tops out at $k = 3.2$ and its bootstrap upper end is $3.126$, $0.074$ below the ceiling:
exactly the signature the rule exists to catch. The Pleias-350M $n{=}100$ arm is not quoted anywhere
(the table uses the $n{=}458$ arm at $0.1\%$), so only KL3M-1.7B needs the extension.

**Recorded before it runs, from the already-committed `results/onset_ci.csv`:** the current reading
is onset $2.5779$, $95\%$ CI $[2.5201, 3.1258]$, ratio $1.1659$ $[1.1397, 1.4136]$, no-crossing
$4.3\%$. Extending to $k \in \{3.5, 4.0, 5.0\}$ on the same 100 passages, same seed, same threshold.
What the Arm A2 precedent predicts: the point estimate barely moves, the no-crossing fraction goes
to zero and the **upper** end widens. The claim in the text is about the **lower** end exceeding
$1$, which an extension upward cannot move; if it does move, that is the finding and the sentence
goes.
