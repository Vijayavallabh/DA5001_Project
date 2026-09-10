# Pre-registration: is the approximation gap an artifact of greedy metering?

Committed before `analysis/marginal_price.py` is run for the first time.

## The question

Theorem 1 lower-bounds the budget any policy needs for a given mean utility by the Cramer rate
function, and the measured decoder exceeds that floor by three to four orders of magnitude. The
paper calls the difference "approximation overhead" and states closing it as the open problem. The
overhead has two candidate sources and they have opposite consequences for a deployer:

1. **The accounting.** Nothing to fix; a metered decoder is simply expensive, and the budget a
   deployer must publish is large for reasons no design will remove.
2. **The decoder.** The audited rule is myopic: it serves the largest `theta` whose charge fits the
   bucket's *current* allowance, at every step, whether or not tilting toward the risky model buys
   anything there. A decoder that allocated the same total budget across the sequence by equalising
   the marginal return per nat would buy more at the same published `k`.

The second is testable offline, with no decoding run and no change to the audited decoder, because
on the geometric path both the charge and the fidelity have closed forms in `psi(u) = log Z(u)`:

    C(theta) = theta psi'(theta) - psi(theta),   C'(theta) = theta psi''(theta)
    G(theta) = theta m - psi(theta),             G'(theta) = m - psi'(theta),  m = psi'(1)

`G(theta) = -D(p_r || p_theta)` up to a per-step constant, so it is fidelity to the risky model,
which is what a metered decoder is trying to buy. Greedy fixes `theta` by the allowance; the optimal
allocation of a fixed total equalises `G'/C'` across steps at one shared price `lambda`. Note
`G'(0) = m - psi'(0)` is the Jeffreys divergence between the two models, the quantity that governs
the exact reward-KL frontier (Monteiro Paes et al.), so the first nat spent at a step is worth its
Jeffreys divergence and the last is worth nothing.

## The measurement

For each protected passage, walk both models along the target teacher-forced, take greedy's per-step
charge at allowance `k` and total it, then find the `lambda` whose equal-price allocation spends that
same total. Report

    gain_ratio = fidelity(equal price) / fidelity(greedy),  at matched total spend

on a grid `k` in {0.25, 0.5, 1.0, 3.0}. The small budgets are the ones that matter: He et al.'s own
appendix says that at `k = 3.0` the constraint is rarely binding, and where `theta = 1` almost
everywhere greedy is trivially optimal because no allocation can exceed the ceiling.

## Bands, fixed now

| `gain_ratio` at the budgets where the constraint binds | reading |
|---|---|
| >= 1.5 | the overhead is substantially a decoder artifact. Build the dual decoder, run it end to end, and the paper gains a constructive answer to its own open problem |
| 1.1 - 1.5 | a real but modest gap; report it as a bound on what redesign buys, do not build the decoder |
| < 1.1 | **greedy is near-optimal in this family.** The open problem has a negative answer: the overhead Theorem 1 exposes is structural, not a failure of this decoder, and that is worth stating plainly because it removes the obvious escape route from the paper's conclusion |

**A prediction that would embarrass the whole direction, recorded so it cannot be quietly dropped:**
if `theta_greedy_at_one_pct` is above 90% at every `k` on the grid, then the constraint never binds
on these passages and the probe measures nothing at all. That is a null from the design, not from
the hypothesis, and it means the grid must move down rather than the conclusion moving.

## What a large gain_ratio would NOT establish

Fidelity to `p_r` is not judged utility, and a decoder that buys more fidelity per nat would also
buy more *extraction* per nat, since reproducing a memorised passage is exactly the behaviour
`p_r` carries. Both halves must then be measured end to end before anything is claimed. If the
reallocation helps extraction at least as much as it helps utility, the finding is not "the
mechanism can be fixed" but something sharper -- that the published `k` fails to determine leakage
even between decoders matched on utility, which is the concession
`sections/appendix_robustness.tex` currently makes.
