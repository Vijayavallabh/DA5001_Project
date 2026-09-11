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
