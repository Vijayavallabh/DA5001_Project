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

## Scored, secondary: selection does not leak, at any $n$ up to $64$

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py --risky-model output/memorizing_llama8b \
    --n-values 1 2 4 8 16 32 64 --limit 100 --max-new-tokens 200 --batch-size 24 --out results
```

```
       arm  KL nats  nv-recall   max    LCS words  >=0.01
     n = 1      0.0   0.0000  0.0000       1.51    0.0%
     n = 2   0.1931   0.0000  0.0000       1.70    0.0%
     n = 4   0.6363   0.0000  0.0000       1.88    0.0%
     n = 8   1.2044   0.0000  0.0000       1.94    0.0%
    n = 16   1.8351   0.0000  0.0000       1.94    0.0%
    n = 32   2.4970   0.0000  0.0000       1.89    0.0%
    n = 64   3.1745   0.0000  0.0000       1.78    0.0%
risky alone       --   0.4338  0.8233         --   80.0%
```

The committed band was $\le 0.005$ at $n = 64$, and the measurement is $0.0000$ with a maximum over
all $100$ passages of $0.0000$: **the first outcome fires**, by the whole width of the band. The
mandatory baselines are on the same passages and the same seeds: the memorising model alone reaches
$0.4338$ mean and $0.8233$ maximum near-verbatim recall, on $80$ of the $100$ passages at or above
$0.01$; $n=1$ *is* the anchor-alone baseline and reaches nothing.

**What the selector's pressure is actually worth.** This is the arm where the pre-registration
expected trouble, because the score being maximised is the *memorising* model's own likelihood, so
the rule is searching $n$ anchor samples for whichever is nearest the protected text. Its whole
effect is about half a word of longest common substring --- the LCS mean runs $1.51$, $1.70$,
$1.88$, $1.94$, $1.94$, $1.89$, $1.78$ across $n$ --- and it is **not monotone**: past $n=8$ more
candidates make the served text slightly *less* similar to the target, because a per-token mean
likelihood rewards fluent continuations rather than the memorised one in particular. Nothing
approaches the $20$-word span near-verbatim recall requires.

That is the bound behaving as Proposition~1 says it must. $P_q(E) \le n\,P_{p_s}(E)$ multiplies a
base rate the $n=1$ row measures at $0.0000$; multiplying it by $64$ leaves $0.0000$. The
certificate is not merely true, it is operative: the protection here is the anchor's support, and
selection cannot reach outside it.

## Scored, primary: the committed arm is REFUTED, and the oracle says exactly why

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 8 \
    --cap-neutral 200 --cap-creative 150 --cap-factual 150 --max-new-tokens 200 \
    --output-dir output/phase5/sel_anchor8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_decoding.py --gen-dir output/phase5/sel_anchor8 --out results
```

```
rule                      n  KL nats  served tokens    u      95% CI        win%
per-token mean (primary)  1   0.0000      98.0       0.3190 [0.281,0.358]   26.4
per-token mean (primary)  2   0.1931      97.9       0.3270 [0.291,0.365]   28.2
per-token mean (primary)  4   0.6363      83.0       0.2900 [0.253,0.327]   24.2
per-token mean (primary)  8   1.2044      69.4       0.3130 [0.279,0.353]   26.6
summed log-likelihood     8   1.2044      15.3       0.2710 [0.238,0.307]   21.8
oracle: the judge itself  8   1.2044     101.2       0.8070 [0.771,0.839]   77.6
```

**The pipeline check passes first.** $n=1$ is the anchor alone and reads $0.3190$ $[0.281, 0.358]$
against the $u_{\text{safe}} = 0.323$ on record from a different judging run of a different sample.
Nothing below would count if that had missed.

**$u_B(8) = 0.3130$, below the committed $0.396$: the arm is REFUTED.** Selecting among eight anchor
samples by the risky model's own per-token likelihood buys *nothing* --- $0.319 \to 0.313$, with
every interval overlapping every other, and not even monotone in $n$. The summed rule is worse
still at $0.271$, and its length column says why: the text it serves collapses from $98.0$ tokens to
$15.3$, which is the pathology the addendum predicted before either rule was scored and is the
reason the per-token mean was made primary.

**The oracle arm is what the pre-registration committed it for, and it separates the two readings.**
Selecting by the judge's own verdict reaches $u = 0.807$ at the same $1.204$ nats --- far above the
metered decoder's best of $0.5015$ at $171.3$ nats, and above the null arm's $0.481$. So the
capacity is there and the budget is not the obstacle: *selection from the anchor's own support can
reach utility the metered decoder never reaches, for a hundred-and-fortieth of the divergence.* What
fails is the signal. The oracle is scored by the judge that selected it and is therefore an upper
bound, not an achievable number; the arm below tests it with a judge that did not.

### Why the risky model's likelihood is a useless selector, measured

Within each prompt, over all $5{,}687$ pairs of candidates the judge ranked differently:

```
within-prompt AUC   the risky model's per-token log-likelihood   0.5258
                    the risky model's summed log-likelihood      0.4777
                    the completion's length in tokens            0.5368
corr(per-token log-likelihood, length) = -0.20
mean length: win 100.9 tokens, tie 101.3, loss 96.8
```

The signal is $0.526$ against a chance of $0.500$: **the likelihood the audited mechanism spends its
entire budget moving toward is a worse predictor of judged quality than the length of the
completion.** The summed rule is *below* chance because it is a length preference in disguise, and
the judge mildly prefers longer text.

That is a second, independent demonstration of what
Appendix~\ref{app:proofs} argues from the scheduling side. The metered decoder's $165$ nats are not
inefficiently spent on utility; they are efficiently spent on a target that is not utility. Give the
same target to a mechanism that spends $1.2$ nats instead of $165$ and it buys exactly as little.
