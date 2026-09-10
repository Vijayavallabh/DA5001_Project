# A budget a deployer can actually compute

**Not pre-registered**, and said plainly because everything else in `results/` was: this is a
reanalysis of `results/anchor_scaling_summary.csv`, which was committed long before, not a new
sweep. What *was* fixed before the numbers were read is the scoring rule --
`analysis/proxy_budget.py` was written with all three candidate predictors and the
leave-one-anchor-out protocol in place, and run once.

## The gap it closes

Section 3 tells a deployer to publish `k/s(x)` and then concedes, two sentences later, that only a
rate quoted as a fraction of ordinary traffic is "available to a deployer who does not know the
protected works in advance". A rights-holder's work is precisely what a deployer has not seen, so
the prescription as it stands cannot be executed.

## The result

Predicting a held-out anchor's surprisal rate on the protected passages from its rate on 50
public-domain Gutenberg texts, scored on identical spans, with one constant fitted on the other
nine anchors:

| predictor | LOO mean abs. error (nats/char) | vs the constant |
|---|---|---|
| public-domain proxy, rescaled | **0.0518** (5.6% of the protected rate) | **4.13x better** |
| a constant, no rescaling | 0.2138 | --- |
| `c_use`, rescaled | 0.4020 | 1.9x **worse** |

Across ten anchors whose protected rate spans 1.86x. The ratio `s_protected / s_proxy` is
1.141 +- 0.077 (cv 6.75%, range 1.060-1.287).

So a deployer can estimate the quantity the budget should be quoted in, to about 6%, from text they
are free to hold. Against an onset that sits at 0.878-0.926 of `s(x)` at matched context, a 6% error
in the unit is comparable to the spread in the ratio itself.

## Two things this does not say

**`c_use` is not a substitute.** It does worse than not rescaling at all. Section 3 does not claim
otherwise -- it says the *margin* `s(x)/c_use` is the reading available without the protected works
-- but the distinction is worth making explicit, because the margin tells a deployer whether a
useful budget exists and does not tell them where to put it.

**The 1.141 is a genre-and-era gap, and it is family-structured**: 1.06-1.09 on the KL3M anchors,
1.13-1.16 on Pleias, 1.25-1.29 on Comma. Fitting per family would be tighter and is not available,
because knowing the family constant requires the protected works. The single constant is the honest
rule and 5.6% is its honest error. The Gutenberg texts are older public-domain books and the
protected works are modern genre novels; a deployer whose protected corpus is not prose fiction has
no reason to expect this constant to carry, and we have not tested one who is.
