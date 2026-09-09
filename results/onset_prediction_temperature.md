# Pre-registration: move s(x) within a single pair and see whether the onset follows

Committed 2026-09-10, before `output/phase5/warp_t0.4_kl3m520m` or `warp_t0.7_kl3m520m` decoded a
token. Only the three budget paths exist, and a budget path needs the anchor alone.

## Why this arm exists

Across the seven pairs, `s(x)` spans 1.61x, and over that range predicting a held-out pair's onset
as `c*s(x)` beats predicting a constant number of nats by only 1.1x in mean absolute error
(0.264 against 0.295 nats, `results/matched_context.csv`). Conditioning on the adversary's context
sharpens it to 5.2x on the five seed-matched pairs, but that grouping is read off the same seven
measurements. Nothing so far moves `s(x)` with everything else held fixed.

The decoder warps both logit vectors with the sampling temperature before the KL solve
(He et al. App. B; `a_patch/factory.py:899-901`), so temperature is a lever on the anchor's
surprisal of the protected work that changes no model, no tokenizer, no corpus and no seed. On
`KL3M-520M + mem. KL3M-520M` it moves `s(x)` by 1.67x on its own -- as much as the whole
ten-model set spans -- and 0.7 is the temperature He et al. run their book experiments at.

     tau     s(x)   k_crit   k_crit/s(x)
     0.4    4.034    6.978         1.730
     0.7    2.689    4.417         1.643
     1.0    2.415    3.791         1.570   <- measured onset 2.4923, ratio 1.0321

## The three accounts and what each predicts

(U) **units**: the onset is a fixed fraction of `s(x)`, this pair's own 1.0321.
    tau 0.7 -> onset 2.775 nats.   tau 0.4 -> onset 4.163 nats.
(N) **null**: the onset is a constant number of nats, this pair's own 2.4923, and dividing by
    `s(x)` is bookkeeping.
    tau 0.7 -> 2.492.   tau 0.4 -> 2.492.
(K) **token bucket**: the onset is a fixed fraction of `k_crit`, this pair's own
    2.4923/3.7912 = 0.6574 (the rule that already predicted both seed arms inside their intervals).
    tau 0.7 -> 2.904 (ratio 1.080).   tau 0.4 -> 4.588 (ratio 1.137).

## Committed prediction

The tau = 0.4 arm decides between (N) and the other two, which differ from it by 67%:

  - onset in **[3.6, 4.6] nats** (ratio 0.89-1.14): (N) is refuted; the onset follows `s(x)`.
  - onset **below 3.0 nats**: (U) is refuted *within a pair*, which is the strongest form of the
    claim, and the paper must say the units result is a cross-pair regularity only.
  - 3.0-3.6: undecided, reported as such.

(U) and (K) sit 10% apart at tau = 0.4 and the bootstrap interval on 100 passages is about +/-0.11
in the ratio, so this arm is **not** expected to separate them; both will be reported.

## Recorded in advance, against ourselves

Warping sharpens the risky model too, so the memoriser at tau = 0.4 is not the memoriser at
tau = 1.0. The k = -1 baseline is therefore reported attached to every point, as it was for the
seed-10 arm, and the standing admissibility screen applies: a sampled k = -1 recall below 0.10
makes the arm inadmissible and it will be reported as inadmissible rather than dropped. If the
baseline moves a lot while the onset ratio does not, that is evidence *for* the law, since the law
does not mention the risky model; if both move together, the arm cannot separate them and will be
reported as confounded.

## Note added while the tau = 0.4 arm was decoding, before any budgeted point was measured

Its `k = -1` baseline came in at nv-recall **0.904**, against **0.519** for the tau = 1.0 control.
Warping sharpens the memoriser as much as it sharpens the anchor, and the arm is admissible by a
wide margin. Recorded now because the direction matters: a memoriser that is *better* at the work
should make extraction begin **earlier**, not later, so the confound pushes the measured onset
**down**, away from (U)'s $4.163$ and towards (N)'s $2.492$. If the onset nevertheless lands near
$4.163$, the confound cannot be what produced it.

## Second pair, committed before either of its arms was launched (2026-09-10)

`KL3M-520M` is a short-context pair whose control ratio is above $1$. Replicating on
`Pleias-1.2B + mem. Pleias-1.2B` -- a seed-matched pair whose control ratio is $0.866$, in the
middle of the five-pair band -- tests the same thing on the other side of the split. Its budget
paths were measured first and are committed with this file; no token of either arm has been
decoded.

     tau     s(x)   k_crit   k_crit/s(x)
     0.4    5.200    9.002         1.731
     0.7    3.513    5.752         1.637
     1.0    3.209    4.942         1.540   <- measured onset 2.7797, ratio 0.8661

At tau = 0.4 this pair's `s(x)` reaches **5.200** nats per token, above the whole built range's
maximum of 3.554, so the two pairs' warped arms together widen the measured range from 1.61x to
2.35x -- which is the reason for running them.

Predictions on the primary metric (`lcs_word >= 4`), each calibrated on this pair's own control:

     arm        (U) units    (N) constant nats    (K) token bucket
     tau 0.7        3.043                2.780               3.235
     tau 0.4        4.504                2.780               5.064

**Committed prediction.** The tau = 0.4 arm again decides between (N) and the other two, which
differ from it by 62% and 82%:

  - onset in **[4.0, 5.3] nats**: (N) is refuted on a second pair, in a second family, at an
    `s(x)` no unwarped pair reaches.
  - onset **below 3.2 nats**: (U) is refuted within a pair and the units result is a cross-pair
    regularity only.
  - 3.2-4.0: undecided, reported as such.

(U) and (K) sit 12% apart here against 10% on the KL3M pair, still inside the bootstrap width at
$n=100$; both are reported. The same admissibility screen and the same baseline-attached reporting
apply: the tau = 0.4 memoriser will be sharper than the control's, which pushes the measured onset
down, against the hypothesis.

## Scored: the KL3M-520M tau = 0.4 arm (2026-09-10)

`output/phase5/warp_t0.4_kl3m520m`, 100 works, twelve budgets, `lcs_word >= 4`.

     quantity            tau 1.0 (control)    tau 0.4     change
     s(x)                          2.4147      4.0340     +67.1%
     k_crit                        3.7912      6.9780     +84.0%
     onset                         2.4923      3.6030     +44.6%
     onset / s(x)                  1.0321      0.8932
     k = -1 baseline                0.519       0.904

**(N), the constant-nats null, is refuted within a single pair.** It predicted 2.492 and the
measurement is **3.603** nats, inside the committed [3.6, 4.6] band that was defined as refuting it,
though at that band's lower edge. The elasticity $d\log(\text{onset})/d\log s(x)$ is
**+0.72 [+0.41, +0.84]**, and the interval excludes 0 by a wide margin. Nothing about this pair
changed except the warp: same anchor, same memoriser, same corpus, same tokenizer, same seed.

**(U), exact proportionality, is not confirmed either.** It predicted 4.163 and the elasticity
interval excludes 1. The onset moves with the rate the budget is charged against, and moves less
than one-for-one.

**The confound points the right way, as registered in advance.** The tau = 0.4 memoriser is much
better at the work --- $k=-1$ recall 0.904 against the control's 0.519 --- and a better memoriser
makes extraction begin *earlier*. So the measured elasticity is a **lower bound** on what it would
be at a fixed memoriser, and the direction of the bias is away from the result, not towards it.

**(K), the token-bucket account, fails here, and that is the informative part.** Calibrated on this
pair's own control it predicted a ratio of 1.137 against a measured 0.893 [0.76, 0.95] --- the first
miss after three straight hits on the seed arms (1.021/1.004, 1.149/1.164, 0.963/0.939). The
asymmetry is not noise and it is interpretable: the seed changes *which* tokens fall in the
adversary's window, which is exactly what a running maximum over the target's surprisal profile
tracks, whereas the warp rescales the whole profile, which the running maximum over-reads. `k_crit`
is a good predictor of what the evaluation protocol does to the onset and a poor one of what the
decoder's own temperature does. Section~\ref{sec:onset} must not present it as a general law.

## Scored: the Pleias-1.2B tau = 0.4 arm, and the replication holds (2026-09-10)

     quantity            tau 1.0 (control)    tau 0.4     change
     s(x)                          3.2094      5.1999     +62.0%
     k_crit                        4.9422      9.0023     +82.2%
     onset                         2.7797      3.7400     +34.5%
     onset / s(x)                  0.8661      0.7192
     k = -1 baseline                0.909       0.952

Measured onset **3.740** nats against the committed bands: [4.0, 5.3] refutes (N), below 3.2 refutes
(U), 3.2-4.0 undecided. It lands in the **undecided** interval, so on the band as written this arm
decides nothing. On the statistic that actually separates the accounts it does:

     pair                elasticity d log(onset) / d log s(x)
     KL3M-520M                        +0.72  [+0.41, +0.84]
     Pleias-1.2B                      +0.61  [+0.40, +0.96]

Two pairs, two families, both intervals excluding **0** --- a constant number of nats is refuted
twice --- and both excluding **1**, so the onset moves with the rate the budget is charged against
and moves less than one-for-one. The point estimates agree and the intervals overlap almost
entirely. Both arms carry the same confound in the same direction: the warped memoriser is better at
the work (0.952 against 0.909 here, 0.904 against 0.519 on KL3M), which pushes the onset down, so
both elasticities are lower bounds.

**The split in what `k_crit` predicts is now measured on both sides.** Scored apart, as
`analysis/seed_effect.py` now reports them:

     intervention                      arms    k_crit mean |rel. err|    no-change null
     seed (s(x) held fixed)               4                    4.0%              9.3%
     temperature (s(x) moved)             2                   31.3%             18.0%

On the seed arms `k_crit` beats the null by 2.3x; on the temperature arms it is **worse than
assuming nothing changed**. Pooling the six gives 13.1% against 12.2% and hides exactly this. The
running maximum tracks which tokens the adversary's window starts on; it over-reads a rescaling of
the whole surprisal profile. Section~\ref{sec:onset} may use it for the first and must not for the
second.

## How the tau = 0.7 arms will be treated, committed before either is scored (2026-09-10)

The tau = 0.7 arms move `s(x)` by only $9.5\%$ (Pleias, 3.2094 -> 3.5134) and $11.4\%$ (KL3M,
2.4147 -> 2.6890), against $62\%$ and $67\%$ at tau = 0.4. In log units that is $0.09$ and $0.11$,
barely twice the $0.05$ guard below which `analysis/seed_effect.py` refuses to report an elasticity
at all. A small denominator makes the estimate unstable, and the partial grid already suggests
Pleias tau = 0.7 will come in near an onset of $2.8$ against its control's $2.780$ -- an elasticity
near zero, where the same pair gave $0.61$ at tau = 0.4.

Committed now, so the treatment is not chosen after seeing whether it flatters the result:

  - Both tau = 0.7 elasticities are reported **with their intervals and with the lever size next to
    them**. A lever of $0.09$ log units is stated wherever the number is.
  - They are **not pooled** with the tau = 0.4 arms into a single mean elasticity. Two doses of
    very different size do not average.
  - If an interval spans both $0$ and that pair's tau = 0.4 estimate, the arm is reported as
    **uninformative about the elasticity**, not as contradicting it. An interval that excludes the
    tau = 0.4 estimate is a genuine inconsistency and will be reported as one, and would mean the
    response is not a constant elasticity and the summary "the onset follows $s(x)$
    sub-proportionally" is too simple.
  - The tau = 0.4 arms remain the ones that carry the claim, because they are the only ones whose
    lever is large enough to measure against.

## Scored: Pleias-1.2B tau = 0.7 --- uninformative, by the rule committed before it ran

     arm                    lever, log units    onset    elasticity
     Pleias-1.2B tau 0.4                0.48    3.740    +0.61  [+0.40, +0.96]
     Pleias-1.2B tau 0.7                0.09    2.734    -0.18  [-1.07, +2.58]

The interval is $3.65$ units wide, and it spans **both** $0$ and this pair's tau = 0.4 estimate of
$0.61$. By the rule committed before the arm was scored, it is therefore reported as
**uninformative about the elasticity**, not as contradicting it. The point estimate alone,
$-0.18$, would read as a contradiction; the lever is $0.09$ log units in $s(x)$ against $0.48$ at
tau = 0.4, and dividing a bootstrap interval on the onset by a denominator that small is what
produces the width. This is the arm the pre-commitment existed for.

What it does establish, with no elasticity needed: the onset fell from $2.780$ to $2.734$ while
$s(x)$ rose $9.5\%$, so at this lever size the measurement cannot distinguish "the onset followed
$s(x)$" from "the onset did not move". Only the tau = 0.4 arms have a lever large enough to
separate those, and they are the ones the claim rests on.
