# Equalise what the evaluation hands the adversary, and see how much of the split is left

**2026-09-11, committed before the two new arms were swept and before any of the nine pairs was
scored in this pipeline.**

The paper's largest open question is stated in its own Limitations: the onset ratio moves with the
words a fixed $20$-token seed buys the adversary, we can *rank* the nine pairs by that quantity at
Spearman $-0.958$, and five interventions move single pairs across the boundary --- but seed words
is $20\times$ characters per token by construction, so the ranking is observational and no choice of
anchors can separate context from granularity. What can separate them is an intervention applied to
*every* pair at once: give each pair's adversary **the same number of words** instead of the same
number of tokens, and ask how much of the spread in onset$/s(x)$ survives.

## What is already on record, and is therefore not a prediction

Two of the nine pairs already have a matched arm, run and reported for a different reason
(Appendix~\ref{app:seed}, `results/seed_effect.csv`): KL3M-520M and KL3M-1.7B at
`--seed-tokens 40`, which buys $14.80$ words against their $7.45$ at $20$ tokens. Their onset ratios
move from $1.032$ to $0.939$ and from $1.155$ to $0.979$. Five more pairs are already at
$13.86$--$15.02$ words at `--seed-tokens 20`, so for them the matched arm **is** the sweep the paper
already reports. So seven of the nine matched ratios exist before this file does, and the spread
they imply can be worked out from numbers already printed in the manuscript. **That part is a
recomputation, not a prediction, and is reported as one.**

What does not exist is (i) the two open-calm pairs at a matched seed, and (ii) any of the nine scored
in one pipeline, on one metric, with one bootstrap. Those are what the bands below are for.

## The blind part: two arms, committed before either was swept

`cyberagent/open-calm-1b` and `cyberagent/open-calm-3b` cut these passages at $2.71$ characters per
token, so at $20$ tokens their adversary holds $9.51$ words --- between the two families, which is
why they were built. At `--seed-tokens 30` the same measurement (`analysis/seed_effect.py:seed_words`,
first $30$ tokens of the same $100$ passages) gives **$80.3$ chars, $14.63$ words**, inside the
coarse family's $13.86$--$15.02$ and beside the KL3M matched arms' $14.80$. That is the arm.

Scored on the **longest common substring in words** at the threshold `analysis/seed_effect.py`
already uses, because a changed seed changes the target's length and an absolute count cannot be
inflated by a shorter target --- the same choice, for the same reason, as the five seed interventions
in Appendix~\ref{app:seed}. The near-verbatim ratio is computed in the same pass and reported beside
it, never instead of it.

| outcome for the two open-calm arms | reading |
|---|---|
| both land in $[0.85, 0.96]$ | handed a coarse-family context they behave like coarse-family pairs; the context is what their $9.5$-word ratios were reporting |
| either lands $\ge 1.00$ | refuted at that pair: a $14.6$-word context does not buy it a coarse-family ratio, so what distinguished it was not the context |
| either lands $< 0.80$ or $> 1.20$ | off the scale of every arm measured, at either seed. A fact about that anchor, reported and not built on |

## The aggregate, committed now

Let $S_{20}$ be the spread (max $-$ min) of onset$/s(x)$ over the nine pairs at `--seed-tokens 20`,
the convention the benchmark actually uses, and $S_{\mathrm{match}}$ the spread over the same nine
when every adversary holds $13.6$--$15.0$ words. Both computed in one pipeline, on the same metric,
with the same bootstrap.

| outcome | reading |
|---|---|
| $S_{\mathrm{match}} \le 0.5\,S_{20}$ | the adversary's context is the **dominant** cause of the split, and most of what looked like a property of the anchors is a property of the evaluation's seed convention |
| $S_{\mathrm{match}} \ge 0.8\,S_{20}$ | **refuted**: equalising the context does not close the split, and the ranking in Section~\ref{sec:onset} is reporting something else that moves with granularity |
| between | partial, and reported as partial with the fraction stated |

**Not a claim of no residue.** The paper already says the interventions do not close the gap between
the families, and nothing here is expected to take the spread to zero. What is being measured is the
*fraction*, on all nine pairs at once rather than one pair at a time.

## Rules, unchanged from every arm before this one

**Entry gate.** A new arm enters only if its *unconstrained* risky model still reproduces the
passages at sampled $k=-1$ recall $\ge 0.10$ on the same $100$ passages and seeds. And the rule
feat-064 committed and applied to the KL3M seed-10 arm stands: if an arm **halves** its control's
unconstrained recall, the seed has weakened the memoriser rather than only moved the budget, and the
arm is reported with that baseline attached and excluded from the aggregate.

**Grid, as a rule and not as numbers.** `analysis/budget_path.py` on the **anchor alone** at
`--seed-tokens 30`, which needs no memoriser and no decoding, then `analysis/grid_from_sx.py`:
eleven budgets at $k/s(x) \in \{0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.15, 1.30, 1.55\}$
rounded to two decimals, plus $k=-1$ and $k=0$. $s(x)$ is re-measured under the new seed because the
seed changes which tokens are the target.

**Bootstrap.** The no-crossing fraction is reported for every arm, and any arm above $1\%$ has its
grid extended upward by the rule already in force.

**Excluded, and named now so they cannot be adopted later.** No re-weighting of pairs, no dropping of
an arm for being inconvenient, no second matched word count chosen after seeing the first, and no
substitution of the near-verbatim metric for the substring metric if the substring answer is less
tidy. One matched target ($13.6$--$15.0$ words), one primary metric, both fixed here.
