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

| $n$ | served the same, host | batch | precision | mean margin (nats) | reward given up, all prompts |
|---|---|---|---|---|---|
| $1$ | $1.000$ | $1.000$ | $1.000$ | --- | $0.000$ |
| $2$ | $0.996$ | $0.994$ | $0.998$ | $8.245$ | $0.0001$ |
| $4$ | $0.982$ | $0.990$ | $0.982$ | $6.408$ | $0.0025$ |
| $8$ | $0.980$ | $0.990$ | $0.978$ | $5.941$ | $0.0049$ |
| $16$ | $0.918$ | $0.936$ | $0.900$ | $5.777$ | $0.0210$ |
| $32$ | $0.906$ | $0.938$ | $0.900$ | $5.452$ | $0.0293$ |
| $64$ | $0.914$ | $0.944$ | $0.918$ | $5.113$ | $0.0300$ |

**Three readings, and the third is the one that matters.**

1. **The served completion is not reproducible, and it gets worse with $n$.** At $n \le 8$ the two
   caches agree on $98\%$ of prompts; by $n=16$ that is $90$--$94\%$ and it then **plateaus** rather
   than continuing to fall. *Falls and then flattens* is the shape; **it is not monotone** --- every
   one of the three series ticks UP from $n=32$ to $n=64$ --- and the wording must not be upgraded to
   "monotone" without re-reading the table (caution (ao)).
2. **The mechanism is the margin, and that IS monotone.** The mean gap between the best and
   second-best reward falls at every step, $8.245 \to 6.408 \to 5.941 \to 5.777 \to 5.452 \to 5.113$.
   Drawing more candidates makes close calls more likely, and a close call is what rounding decides.
3. **The quality served is reproducible even where the text is not.** The alternative pick gives up
   **$0.030$ nats** at $n=64$ averaged over all prompts, and only $0.35$ nats averaged over the
   prompts that actually disagree --- against a winner-to-runner-up margin of $5.1$ nats and rewards
   near $-27$. **Two caches that disagree about which candidate wins agree about what it is worth.**
   So this is a reproducibility property of the *served text*, not a utility defect.

**And precision, not hardware, is the variable.** Changing only the numerical precision on ONE
machine agrees on $0.900$ of cells at $n=16$, where changing the GPU architecture agrees on $0.918$.
Whatever the second host costs, bf16 costs at least as much without leaving the machine. This is the
same conclusion `feat-136`'s within-host controls reached from the reward distribution, reached again
from the quantity a deployer actually sees.

## What this does not show

It compares one anchor, one reward model and one prompt set, and the $500$ prompts are the committed
selection corpus. It says nothing about whether a *judged* gain moves --- two texts of equal reward
need not be of equal judged utility, and nothing here judges anything. It does not establish that
fp32 scoring would remove the effect; the arms that would answer that are running and are reported
in `results/onset_prediction_breadth_ladders.md`. And because nothing was registered in advance,
**no verdict here may be promoted into a claim without an arm that commits its bands first.**
