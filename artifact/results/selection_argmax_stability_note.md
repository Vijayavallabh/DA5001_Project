# Is the completion best-of-$n$ SERVES reproducible? --- an arm with **no committed bands**

**This is not a pre-registered arm.** It has **no committed bands**, it was not planned, and it is
deliberately **not** an `onset_prediction_*.md`. It exists because `feat-136`'s instrument gate
failed, and the control that diagnosed the failure turned out to measure something this paper does
not have. Everything here is exploratory and must be read as such; the same convention
`results/selection_alpaca_note.md` already sets.

Zero GPU: every number comes from reward caches already written for other purposes.

## What is measured

Selection anchoring draws $n$ completions from the anchor, scores them with a pointwise reward, and
serves the argmax. Proposition 1 bounds $q(y) \le n\,p_s(y)$ for **any** score and **any** tie rule,
so the certificate is indifferent to which candidate wins a near-tie. A deployer is not: they cannot
reproduce an answer they cannot re-derive. `analysis/selection_argmax_stability.py` asks, per $n$,
how often two caches of the same $32{,}000$ candidate texts serve the same candidate, what the
winner-to-runner-up margin is, and --- the part that decides whether any of it matters --- **what the
other cache's pick gives up, measured on the reference cache's own reward scale.**

Three caches differ in exactly one thing each: the GPU architecture, the batch size (hence the
reduction order), and the numerical precision. The candidate TEXTS are identical in all of them.

## What it says

Four comparisons, each differing from its reference in exactly one thing. "Served the same" is the
fraction of the $500$ prompts whose served candidate is identical under both caches.

| $n$ | host (bf16) | batch (bf16) | precision (bf16 vs fp32) | batch (fp32) | mean margin, nats |
|---|---|---|---|---|---|
| $1$ | $1.0$ | $1.0$ | $1.0$ | $1.0$ | $---$ |
| $2$ | $0.996$ | $0.994$ | $0.998$ | $1.0$ | $8.24513$ |
| $4$ | $0.982$ | $0.99$ | $0.982$ | $1.0$ | $6.40775$ |
| $8$ | $0.98$ | $0.99$ | $0.978$ | $1.0$ | $5.94101$ |
| $16$ | $0.918$ | $0.936$ | $0.9$ | $0.938$ | $5.77688$ |
| $32$ | $0.906$ | $0.938$ | $0.9$ | $0.928$ | $5.45202$ |
| $64$ | $0.914$ | $0.944$ | $0.918$ | $0.956$ | $5.11322$ |

**1. The served completion is not reproducible, and it gets worse with $n$ --- then stops getting
worse.** At $n \le 8$ the bf16 caches agree on $98\%$ of prompts; by $n=16$ that is $90$--$94\%$ and
it then **plateaus**. *Falls and then flattens* is the shape; **it is not monotone** --- every bf16
series ticks UP from $n=32$ to $n=64$ --- and the wording must not be upgraded to "monotone" without
re-reading the table (caution (ao)).

**2. The mechanism is the margin, and that IS monotone.** The mean gap between the best and
second-best reward falls at every step, $8.245 \to 6.408 \to 5.941 \to 5.777 \to 5.452 \to 5.113$.
More candidates means more close calls, and a close call is what rounding decides.

**3. The quality served is reproducible even where the text is not.** The alternative pick gives up
$0.03$ nats at $n=64$ over all prompts, and $0.35$ over the prompts that actually disagree --- against
a margin of $5.1$ nats and rewards near $-27$. Two caches that disagree about which candidate wins
agree about what it is worth. This is a reproducibility property of the served **text**, not a
utility defect, and Proposition~1 holds for any tie rule so the certificate is untouched.

**4. What fp32 does, measured rather than guessed --- and it is not what was expected.** Before the
control ran, the expectation written into `results/onset_prediction_breadth_ladders.md` was that
scoring in fp32 would "restore a reproducible served completion". **It does not, and it does
something more interesting.** In fp32 the *rewards* become essentially exact --- $32{,}000$ of
$32{,}000$ agree within $10^{-2}$, mean $\lvert$diff$\rvert$ $0.00006$, a factor of about $1500$
better than bf16 --- and the served argmax becomes **perfectly** reproducible at $n \le 8$. At
$n \ge 16$ it still moves on $4$--$7\%$ of prompts. **But in fp32 those disagreements cost
$0.00001$ nats, against bf16's $0.011$: a thousandfold difference.** So the two regimes are
qualitatively distinct: in bf16 rounding *overturns a real preference*; in fp32 all that remains is
ties between candidates the reward model genuinely cannot separate. At $n \ge 16$ the top candidates
routinely sit within $10^{-4}$ nats of each other, which is a statement about the **scorer
saturating**, not about arithmetic.

**5. Precision and host are the same order of magnitude, and neither dominates.** An earlier draft of
this note claimed precision was "at least as disruptive as changing hosts". **That is false**, and it
was true only at the $n$ it was asserted at: comparing the precision and host columns above, precision
is worse at $n \in \{8, 16, 32\}$, **better** at $n \in \{2, 64\}$, and equal at $n \in \{1, 4\}$.
The defensible statement is the weaker one, and it is all the argument needs: **changing precision on
one machine moves the served completion about as much as changing the GPU architecture does**, so
"the second host is a different instrument" was never the explanation.

**6. Every disagreement rate here is a LOWER bound.** Rewards are stored rounded to $5$ decimal
places, and $30$ of the $500$ prompts have their top two stored rewards tie **exactly** at $n=64$
--- duplicate or near-duplicate completions. `max` breaks a tie by index, identically in both caches,
so those prompts agree for a reason that has nothing to do with the instrument. Rounding and ties can
only ever make the argmax agree more, never less.

## What this does not show

It compares one anchor, one reward model and one prompt set, and the $500$ prompts are the committed
selection corpus. It says nothing about whether a *judged* gain moves --- two texts of equal reward
need not be of equal judged utility, and nothing here judges anything. It does not establish that
fp32 scoring would remove the effect; the arms that would answer that are running and are reported
in `results/onset_prediction_breadth_ladders.md`. And because nothing was registered in advance,
**no verdict here may be promoted into a claim without an arm that commits its bands first.**
