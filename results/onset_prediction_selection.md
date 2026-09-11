# The construction the no-free-lunch section disclaims. Committed before any of it was generated.

**2026-09-11.** `sections/frontier.tex` measures the mechanism against Theorem~\ref{thm:nfl} and then
says, in as many words, what the measurement does *not* license:

> Eq.~\eqref{eq:nfl} is attained by \emph{some} distribution, not necessarily one anybody would
> serve, so the gap licenses the negative claim --- a metered decoder's budget is not set by how much
> better its output is --- and not the constructive one.

`results/utility_price.csv` puts numbers on the gap: at $k=3$ the decoder spends a mean of $165.0$
nats of realised sequence divergence to move judged utility from the anchor's $0.323$ to $0.469$,
where the Cram\'er rate function says $0.052$ nats would do. The ratio is $3192\times$. The obvious
question --- *is there a decoder anybody would serve that spends $O(1)$ nats?* --- has never been
asked of this workload, and it can be, with the two models the audit already has.

## The mechanism, fixed here before anything is generated

**Selection anchoring.** Draw $n$ completions $y_1 \dots y_n \sim p_s(\cdot \mid x)$ from the anchor.
Score each by $\log p_r(y_i \mid x)$, the *risky* model's likelihood --- the same objective the
anchored decoder tilts toward, computed by the same model, so the two mechanisms consume the same
two checkpoints. Serve $y^\star = \arg\max_i \log p_r(y_i \mid x)$. Nothing else changes: same
prompts, same temperature $1$, same $200$-token cap, same harness.

Two properties, both immediate and neither new as mathematics
(\cite{beirami2025bestofn}, \cite{gao2023scaling}; `background.tex` already states the first):

1. $D_{\mathrm{KL}}(q \,\|\, p_s) \le \log n - (n-1)/n$. At $n=8$ that is $\mathbf{1.204}$ nats.
2. $q(y) \le n\,p_s(y)$ pointwise, so $P_q(E) \le n\,P_{p_s}(E)$ for every event $E$ --- Proposition~1's
   bound with $K = \log n$. The vacuity threshold $K = S(x)$ is therefore reached at
   $n = e^{S(x)}$, and $S(x)$ for these passages is about $850$ nats.

**The structural point, which is what makes this worth running.** A per-token budget is $K = kT$ and
grows linearly with the length of what is generated; a selection budget is $\log n$ and does not
grow at all. On the works in this paper the metered decoder's $k/s(x)$ is length-invariant --- that
is why the onset collapses --- while a selection mechanism's $\log n / S(x)$ *falls* as the work
gets longer. Whether that asymmetry buys anything in practice is an empirical question about how
much utility survives selection from the anchor's own support, and that is what is measured below.

## Committed bands, primary: does it buy the decoder's utility?

$u$ is the judged utility already defined in `analysis/utility_price.py`: a completion scored $1$ for
a win, $\tfrac12$ for a tie and $0$ for a loss against the *unconstrained* risky model's completion
on the same prompt, by Qwen2.5-7B-Instruct with the presentation order randomised per pair. The
comparison points are already on record and are not re-measured: $u_{\text{safe}} = 0.323$ (the
anchor alone), $u = 0.469$ at $k=3$ ($165.0$ nats), $u = 0.5015$ at $k=10$ ($171.3$ nats), and the
null arm --- the risky model judged against itself at another seed --- at $0.481$.

| outcome for $u_B(n{=}8)$ | reading |
|---|---|
| $\ge 0.469$ | **the construction works**: the $k=3$ decoder's judged utility for $1.204$ nats instead of $165.0$, a $137\times$ reduction, with the certificate intact at $\log 8 = 2.08$ against $S(x) \approx 850$ |
| $0.396$ to $0.469$ | **partial**: at least half the decoder's gain for under $1\%$ of its divergence. Reported as partial, with the fraction stated |
| $< 0.396$ | **refuted**: selection from the anchor's support recovers less than half the gain, and the decoder's per-token tilting is buying something selection cannot. That is a result about where the $3192\times$ gap really lives |
| $> 0.481$ | above the null arm, i.e. beating the risky model's own self-agreement. Reported, and read as a statement about the judge and the metric rather than about the mechanism |

$n \in \{1, 2, 4, 8\}$ are all scored, from one judging pass over all eight candidates per prompt, so
every arm reuses the same judgments and none is judged twice. $n=1$ is the anchor-alone control and
must reproduce $0.323$ to within its own interval, or the pipeline is wrong and nothing else counts.

**An oracle arm is computed and is not a result.** Selecting by the judge's own verdict instead of
by $\log p_r$ gives the best any $n=8$ selection could do on this metric. It is reported as a
ceiling, labelled an oracle, and never used to score the bands: it separates ``selection cannot
reach it'' from ``$\log p_r$ is a poor selector''.

## Committed bands, secondary: does selection leak?

The worry is exact and adversarial: the selector maximises the *memorising* model's likelihood, so it
actively searches the $n$ anchor samples for whichever is closest to the protected text. The bound
$P_q(E) \le n P_{p_s}(E)$ says this cannot help much, but the anchor's own recall is $0.000$ only to
the precision measured, and $n = 64$ multiplies it.

Run on the same $100$ protected passages, the same seeds, and with $k=-1$ (risky alone) and $k=0$
(anchor alone) on the same prompts, as every experiment in this paper does.

| outcome for near-verbatim recall at $n=64$ | reading |
|---|---|
| $\le 0.005$ | selection does not find memorised text in the anchor's samples; the $\log n$ certificate is not merely true but operative |
| $0.005$ to $0.02$ | a measurable rise from the anchor's $0.000$, reported with its $n$ and not smoothed over |
| $\ge 0.02$ | **refuted**: selection is itself an extraction attack, and the $\log n$ bound --- which still holds --- is bounding an event whose base rate is not as small as the $n=1$ measurement suggests |

## What this costs, stated before the result is known

$n$ generations and $n$ likelihood evaluations per served response, against the anchored decoder's
one forward pass through each model per token. At $n=8$ that is roughly $8\times$ the inference
compute for the generation and comparable for the scoring. A mechanism that is cheaper in nats and
dearer in FLOPs is a different trade, not a free lunch, and it is reported as one.

## Excluded now, so they cannot be adopted later

No re-tuning of the sampling temperature after seeing the result ($\tau = 1$, as every other arm).
No changing the selection score from $\log p_r$ --- it is fixed because it is the objective the
audited mechanism itself optimises. No reporting the oracle arm as the mechanism. No dropping a
prompt class, and no switching the utility definition away from the three-point judged verdict
`analysis/utility_price.py` already uses. If $n=8$ misses its band, the value of $n$ that would be
needed is reported by extrapolation and is not run as a replacement for the committed arm.

## Addendum, committed while the anchor samples were still generating and nothing had been scored

`log p_r(y \mid x)` above does not say whether the score is summed over the continuation or averaged
per token, and the two are different rules. Summed likelihood is monotonically decreasing in length,
so on this workload --- where the anchor's completions run from a few words to the full $200$-token
cap --- it would select the shortest candidate almost every time, and the experiment would be a test
of length preference rather than of selection.

**Fixed now, before any candidate is scored: the primary selection rule is the per-token mean,
$\frac{1}{|y|}\log p_r(y \mid x)$.** Two reasons, both stated before the numbers exist: the audited
mechanism meters *per token* and its budget $k$ is a per-token rate, so a per-token score is the
matched quantity; and a rule whose answer is ``pick the shortest'' is not a decoder anyone would
serve, which is the standard this construction has to meet.

The summed rule is computed in the same pass and reported beside it, because the difference between
them is informative about what the risky model's likelihood is actually selecting for. It is not
the primary and cannot be swapped in: if the summed rule scores better, that is reported as a
finding about length, not as the mechanism's result.

Nothing else changes --- not a band edge, not $n$, not the utility definition, not the oracle arm's
status as a ceiling rather than a result.
