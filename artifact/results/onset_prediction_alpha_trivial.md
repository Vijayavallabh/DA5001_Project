# Pre-registration: is $\alpha=8$'s $80\times$ safety a repair, or the trivial horn?

Committed **2026-09-17 14:10**, before the arm ran. Nothing above `## Scoring log` is edited after.

## Why

Section~5 calls Repair 1 --- sharpen the accounting, raise the Rényi order at the same published
budget --- a failure, on the ground that Proposition 2 says the vacuity threshold does not move. A
reviewer rejects that as unresponsive, and the objection is strong:

> *"At $k=3$, raising the charge order from $\alpha=1$ to $\alpha=8$ cuts oracle-window attack recall
> from $0.0965$ to $0.0012$ and touches $90.3\%$ of ordinary-traffic steps differently (distinct-3
> $\approx$ unchanged $0.985$--$0.992$). By the paper's own pragmatic standard, $\alpha=8$ is
> $\sim 80\times$ safer at zero measured utility cost. A rights-holder cares about realized risk.
> The paper's own Table 3 is evidence FOR the repair."*

The premise is arithmetically correct and the paper should concede it rather than argue certificate
semantics at someone asking about harm.

But Table~\ref{tab:orders} contains an answer the paper never states, two columns to the right of the
one the reviewer read. At $\alpha=8$ the risky model's own next-token choice survives at **$0.08\%$**
of ordinary steps, against $\alpha=1$'s **$99.3\%$**, and $99.6\%$ of steps are touched. That says
$\alpha=8$ bought its $80\times$ by **becoming the safe model** --- Proposition 3's trivial horn, the
thing the dichotomy names --- rather than by metering better.

**That reading is currently untested.** `risky unchanged` is a mechanism statistic, not a utility one,
and distinct-3 is a fluency proxy that cannot see capability loss: a fluent safe-model completion
scores $0.987$ exactly as a fluent fused one does. The reviewer is right that the paper has not
measured the thing it needs, and right that what it has measured does not answer them.

## What is measured

$D_2^{\alpha=8}$: the order-averaged judged gain of the $\alpha=8$, $k=3$ decoder **over the anchor
served alone** ($k=0$), on the $500$ ordinary prompts, same fixed opponent, through
`analysis/order_averaged_h2h.py` --- the same instrument and protocol as every judged number the
paper quotes.

The existing $\alpha=8$ utility arm (`output/phase4/util_renyi_8`) is **$150$ prompts at one seed**,
too thin against a cross-pass noise floor of about $0.04$. The arm is regenerated at $k=3$ on the
full $500$-prompt ordinary set at the three seeds `sweep_plain` used, so it is judged at the same
power as the headline.

**Reference computed in the same pass:** $D_2^{\alpha=1}$, the KL decoder at the same $k=3$
(`output/phase2/conc_all`), so the comparison is against the deployed rule rather than against zero.

## Committed bands

| $D_2^{\alpha=8}$ reads | verdict |
|---|---|
| $95\%$ interval includes $0$ | **TRIVIAL HORN CONFIRMED.** $\alpha=8$ is not judged-separable from the safe model it exists to improve on; its $80\times$ is the safe model's safety, bought by walking onto the horn Proposition 3 names. Section~5 states this measured fact instead of the threshold argument, and concedes the reviewer's premise in the same breath. |
| $>0$, interval excludes $0$, and not below $D_2^{\alpha=1}$ by an interval excluding $0$ | **REPAIR PARTLY SUCCEEDS.** $\alpha=8$ keeps utility over the safe model *and* cuts extraction $80\times$. "The three repairs fail where the theory says they must" is then **false as written**, and the abstract and Section~5 are corrected: the objection to a higher charge order is confined to certificate semantics, and the realised-risk reading belongs to the reviewer. |
| $>0$ but below $D_2^{\alpha=1}$, the difference's interval excluding $0$ | **PRICED.** $\alpha=8$ buys its $80\times$ by giving up a measured fraction of what $\alpha=1$ buys. Report the exchange rate --- nats of extraction bought per point of judged utility --- and stop calling the repair either failed or successful. |

## Committed secondary

Report $D_2^{\alpha=1}$ and the paired difference $D_2^{\alpha=1} - D_2^{\alpha=8}$ with its interval,
whatever they read. The `risky unchanged` column ($99.3\%$ vs $0.08\%$) is reported beside them as
the *mechanism* statistic it is, never as the utility evidence --- which is the error this arm exists
to stop the paper making.

## Excluded in advance

We will not, after seeing the result: switch to distinct-3, risky-NLL, or any other proxy as the
utility metric; drop $\alpha=1$ from the comparison; change the judge or the opponent; re-run at a
different $k$ and report that instead; or quote the existing $150$-prompt arm if the $500$-prompt one
disagrees with it.

**This arm can cost the paper a claim**, and the claim is in the abstract. If $\alpha=8$ holds its
utility, "the three repairs this predicts fail where it says they must" is withdrawn.

## Scoring, 2026-09-17 17:20 — **TRIVIAL HORN CONFIRMED**, on a knife-edge, and the reviewer's premise survives

Regenerated $\alpha=8$ at $k=3$ on the full $500$-prompt ordinary set at the three seeds
`sweep_plain` used (16:05–16:31, GPU 4, `rc=0`), then judged through
`analysis/order_averaged_h2h.py` with the new `--metered-constraint renyi:8`. The $\alpha=1$
reference is a second pass at the same budget, opponent, control, judge and prompts.

| arm at $k=3$ | gain over the anchor served alone | reading |
|---|---|---|
| $\alpha=1$ (the deployed rule) | $+0.034$ $[+0.0080, +0.0595]$ | separates |
| $\alpha=8$ | $+0.021$ $[-0.0005, +0.0425]$ | **does not separate** |
| paired difference $\alpha{=}1 - \alpha{=}8$ | $+0.0130$ $[-0.0115, +0.0380]$ | **not distinguishable** |

### Primary: band 1, and it is met by $0.0005$

$\alpha=8$'s interval includes zero, so the committed reading is **TRIVIAL HORN CONFIRMED**: at the
same published budget the $\alpha=8$ decoder is not judged-separable from the safe model it exists to
improve on, where $\alpha=1$ is. It is recorded as the knife-edge it is --- the interval clears zero
by $0.0005$ and the point estimate is $+0.021$, not $0$ --- and not as a clean null.

### **The reviewer's premise is not refuted, and that is reported first**

Their claim was that $\alpha=8$ is $\sim 80\times$ safer *at no measured utility cost*. **We could not
measure a cost.** The paired difference against $\alpha=1$ is $+0.0130$ with an interval that
includes zero, so this arm does **not** license the sentence "$\alpha=8$ gives up utility". Three
readings sit inside each other's intervals here (anchor, $\alpha=8$, $\alpha=1$) and the comparison
is underpowered to separate them; what it does establish is narrower: **$\alpha=8$ fails to clear the
bar $\alpha=1$ clears.**

### What does not depend on the judge

The mechanism statistics are unambiguous and carry the argument the judged numbers cannot:

| | $\alpha=1$, $k=3$ | $\alpha=8$, $k=3$ |
|---|---|---|
| ordinary steps where the constraint is active | $0.35\%$ | $\mathbf{99.28\%}$ |
| steps serving the risky model unchanged | $99.35\%$ | $0.17\%$ |
| served text identical to the anchor's own draw | $2/500$ | $21/500$ |
| ROUGE-L against the anchor / against the risky model | $0.076$ / $0.104$ | $0.169$ / $0.144$ |

At $\alpha=1$ the constraint is slack almost everywhere and the decoder *is* the risky model --- which
is what He et al.'s own Appendix~E.1 says of $k=3$. At $\alpha=8$ it binds almost everywhere and the
output leans to the anchor on every measure. **That is where the $80\times$ comes from**: not from
metering the same decoder more finely, but from ceasing to serve the risky model. The certificate is
unchanged --- both publish $K = 3T$, both vacuous for $100\%$ of the protected passages.

### The consequence for the manuscript

Section~5 currently answers this objection with Proposition~\ref{prop:threshold}, that the vacuity
threshold does not move. That is true and does not respond to someone asking about harm. The
replacement concedes the $80\times$, concedes that no utility cost is measurable, and makes the
mechanism claim instead: a higher order does not meter better, it stops serving the risky model, and
at that point it no longer measurably improves on the anchor. The abstract's "the three repairs this
predicts fail where it says they must" survives --- Repair 1 fails on the horn Proposition 3 names ---
but Section~5 must stop resting it on threshold semantics alone.

### Band D honoured

No switch to distinct-3 or risky-NLL as the utility metric; $\alpha=1$ is reported beside $\alpha=8$
rather than zero; judge, opponent and prompts untouched; no re-run at a different $k$; and the
existing $150$-prompt arm is not quoted.

**Deviation, recorded:** the registration said the $\alpha=1$ reference would be computed "in the same
pass". `order_averaged_h2h.py` judges one metered arm per pass, so it is a second pass at identical
settings. That is sound here because judging is deterministic --- `u_anchor_k0` is identical on
$500/500$ prompts across the two passes, checked before the paired difference was formed, so the
shared control cancels exactly.
