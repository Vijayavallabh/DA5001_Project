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
