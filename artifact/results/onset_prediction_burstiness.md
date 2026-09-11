# Is the residue the burstiness the paper's own Proposition 2 names?

**Committed 2026-09-11, before the quantity was computed.** Everything this needs is already in
`results/budget_path_*.csv`, which is why the band has to be written down first: the test costs no
GPU and thirty seconds, so nothing but a commit separates a hypothesis from a story.

## The gap this is aimed at

Section~\ref{sec:onset} reports that the onset ratio takes two values --- $0.878$--$0.926$ on the
five pairs whose tokenizer emits about four characters per token, $1.053$ and $1.166$ on the two
KL3M pairs at about two --- and Limitations says the residue is not resolvable:

> ... none [of our anchors] falls between the two granularity groups --- the gap from $2.4$ to $3.4$
> characters per token is empty across all fourteen models in our cache, so the next test needs a
> tokenizer trained for it rather than chosen from what exists.

Five seed interventions move each family toward the other and none reaches the other's band, so the
adversary's context is part of it and not all of it. Vocabulary size, memorisation strength, target
length and step count are each excluded by their own control. What is left is unexplained.

## The hypothesis, which is the paper's own Proposition 2

Proposition~\ref{prop:outrun} already says a budget publishes a **drift** where safety is a
**workload maximum**, and names the gap: $k_{\mathrm{crit}}(x) \ge s(x) + \delta/N$, with equality
only when the surprisal accumulates at a constant rate. The ratio $k_{\mathrm{crit}}/s(x)$ is
therefore a per-pair measure of how *bursty* the protected work's surprisal process is under that
anchor, and it is already computed, per passage, for all seven pairs.

A finer tokenizer cuts the same text into more, individually less informative tokens, so its
per-token surprisal sequence is a different process --- and a token bucket that meters a mean rate
is exactly the object that handles burstiness badly. If that is the residue, then a pair whose
surprisal is burstier under its own anchor should need its budget raised **further above** $s(x)$
before the work leaks, which is what an onset ratio above $1$ is.

**Predicted sign: positive.** More burstiness, higher onset ratio.

## Committed bands

Over the seven pairs of the main onset table, with $x = \operatorname{med}_p k_{\mathrm{crit}}(p) /
\operatorname{med}_p s(p)$ from each pair's own `budget_path` file (both per token, so the ratio is
dimensionless) and $y$ its onset ratio from `results/onset.csv`:

| outcome | reading |
|---|---|
| Spearman $\rho \ge +0.786$ (exact $p \le 0.048$ over all $7!$ orderings) | the residue is burstiness, and it is Proposition~\ref{prop:outrun}'s own quantity. Section~\ref{sec:onset} stops calling it unexplained and names it |
| $\rho$ between $+0.6$ and $+0.786$, or $\le -0.6$ | inconclusive at seven pairs, reported with its $p$ and not built on. A negative $\rho$ of that size is itself worth stating, since the derivation requires it positive |
| $|\rho| < 0.6$ | refuted. The residue stays unexplained and Limitations keeps saying so |

**Secondary, and the sharper form.** If $k_{\mathrm{crit}}$ is the better normaliser, then
onset$/k_{\mathrm{crit}}$ should be *more* constant across the seven pairs than onset$/s(x)$.
Committed: the coefficient of variation of onset$/k_{\mathrm{crit}}$ against that of
onset$/s(x)$, which is $0.1022$ (`results/onset_units.csv`, both unit systems). A lower CV supports
the primary; a higher one contradicts it even if the correlation passes, and both are reported.

**What will not happen.** No sweep over alternative burstiness statistics until one of them works.
$k_{\mathrm{crit}}/s(x)$ is named in advance because it is the one Proposition~\ref{prop:outrun}
already defines; $s_{\max}/s_{\text{mean}}$, $s_{p90}/s_{\text{mean}}$ and $s_{\text{std}}$ are in
the same CSVs and are **not** part of this test. If the committed statistic fails, the residue is
unexplained and that is the report.

**The standing caution.** $k_{\mathrm{crit}}$ is not a free variable: it is computed from the anchor
alone, it is confounded with the tokenizer exactly as everything else here is, and it cannot
separate granularity from what granularity determines. A pass would say the residue has a *name* and
a computable proxy, not that a causal test has been run. The paper must say that in the same
sentence.

## Scored: refuted, and the refutation is worth more than the hypothesis would have been

```
.venv/bin/python analysis/onset_burstiness.py --out results
```

```
pair                                s(x)   k_crit  k_crit/s  onset/s  onset/k_crit
Pleias-1.2B + mem. Pleias-1.2B     3.209    4.942     1.540    0.878         0.570
Pleias-350M + mem. Pleias-350M     3.554    5.503     1.548    0.895         0.578
KL3M-520M + mem. KL3M-520M         2.415    3.791     1.570    1.053         0.671
KL3M-1.7B + mem. KL3M-1.7B         2.211    3.644     1.648    1.166         0.708
Phi-3.5-mini + mem. Phi-3.5-mini   2.837    5.556     1.958    0.926         0.473
TinyComma-1.8B + mem. Llama-8B     3.239   13.932     4.302    0.887         0.206
Comma-7B + mem. Comma-7B           2.393   11.319     4.729    0.892         0.189
```

**$\rho = +0.036$, exact $p = 0.96$.** Squarely inside the committed refuting band $|\rho| < 0.6$.
The residue is not burstiness as Proposition~\ref{prop:outrun} measures it, and Limitations keeps
saying it is unexplained.

The table shows why, and the reason is not subtle. The two pairs with by far the *most* bursty
surprisal --- TinyComma-1.8B and Comma-7B, at $k_{\mathrm{crit}}/s(x)$ of $4.30$ and $4.73$ against
$1.54$--$1.96$ for the rest --- sit at onset ratios of $0.887$ and $0.892$, in the middle of the
coarse family. The two KL3M pairs, whose ratios exceed $1$, are the *second and third least* bursty
of the seven. The hypothesis does not merely fail to reach significance; the two most extreme points
sit on the wrong side of it.

### The corollary is a positive result about the paper's own unit

The secondary comparison was committed as a check on the primary and answers a question the paper
could not previously answer at all: `results/collapse_robustness.csv` lists
$k_{\mathrm{crit}}$ among the normalisers with `nan` at $n=0$, because its $k/k_{\mathrm{crit}}$
overlap window is empty on that grid. Ranking the seven pairs instead:

```
coefficient of variation across the seven pairs
  onset / s(x)      0.1154
  onset / k_crit    0.4340      3.8x worse
```

So the anchor's surprisal **rate** is the unit, and the running maximum --- which is the right
quantity for *reproducibility*, and is what Proposition~\ref{prop:outrun} is about --- is the wrong
one for *onset*, by a factor of nearly four. The two propositions really do describe different
boundaries, and this is the first measurement that separates them on the same seven pairs rather
than asserting the distinction. Section~\ref{sec:onset} already says $k_{\mathrm{crit}}$ tracks
which token of a fixed profile the adversary must generate first and not a rescaling of the profile;
this is that statement, measured between pairs instead of within one.

**A note on the CV.** $0.1154$ here against the $0.1022$ the pre-registration quoted from
`results/onset_units.csv`: that file uses the Pleias-350M $n{=}458$ arm ($0.920$) and
`results/onset.csv`, which this reads, carries the $n{=}100$ arm ($0.895$). Both CVs above are
computed from the same seven rows, which is the comparison that matters; the $3.8\times$ is
unaffected either way.

**Nothing else was tried.** $s_{\max}/s_{\text{mean}}$, $s_{p90}/s_{\text{mean}}$ and
$s_{\text{std}}$ sit in the same CSVs, were named in advance as excluded, and stay excluded. The
committed statistic failed; the residue is unexplained; that is the report.
