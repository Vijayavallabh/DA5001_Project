# Pre-registration: was the refuted refinement wrong, or only approximated?

Committed **before the measurement runs**. Nothing above the `## Scoring log` line is edited
afterwards.

## The honest status of this arm, stated first

Eq. (req), `r(x) = s_s(x) - s_r(x)`, was committed for the last five pairs **before any of them was
swept** and is refuted by all three of its rules. That verdict stands and is not under review here.

This arm is **a diagnostic, not a prediction**: the nine onsets are already measured and on record,
so nothing tested below is blind. It is worth running anyway because it separates two very
different explanations of the same refutation, and the paper currently asserts one of them without
evidence:

* **the approximation failed** --- `r(x)` is the *near-deterministic collapse* of the quantity the
  geometry actually names, which is the full-distribution mean
  `kbar(x) = (1/T) sum_t D_KL(p_r,t || p_s,t)` along the target's own prefixes. `r(x)` evaluates
  that integrand at the single realised token. Where the memoriser is not near-deterministic the
  two differ, and every miss on record is low, which is what a systematically biased estimator
  looks like; or
* **the quantity is the wrong one** --- the budget granting access to a *class* of works is not the
  budget at which one work becomes recoverable, which is what the manuscript says now.

Whichever it is, the sentence in Appendix~C should say it for a measured reason. Because this is
not blind, no reading below may be written up as a confirmed prediction, and the bands exist only
so the analysis cannot be steered after the numbers are seen.

## What is measured

For each of the nine (anchor, memoriser) pairs, on the **same passages** the onset used (the
`works()` selection of `analysis/onset_theory.py`, `--limit 100`, `attack_train`/`val`/`test`
intersected with the pair's `budget_path` prompt ids -- the memorisers saw `attack_train` and
`val`, caution (h)), with both models teacher-forced over prefix + target and the tokenizer shared
by construction:

* `kbar(x)` -- mean over target positions of the exact `D_KL(p_r,t || p_s,t)` over the full
  vocabulary, in nats per token;
* `r(x)` -- the plug-in `log p_r(x_t) - log p_s(x_t)` averaged over the same positions, recomputed
  in the same pass so the comparison is like-for-like rather than against a stored column.

## Bands, committed before the run

**E1 -- is the plug-in below the exact rate, and how often.** Fraction of passages with
`kbar(x) > r(x)`, pooled over the nine pairs.

| reading | band |
|---|---|
| BIASED LOW | >= 0.90 |
| MIXED | 0.60 -- 0.90 |
| NOT THE EXPLANATION | < 0.60 |

**E2 -- does the exact rate predict the onset better than the plug-in.** Per pair, the relative
error of the median rate against the measured onset, `|median rate / onset - 1|`; compare the
median of that over the nine pairs for `kbar` and for `r`, and the sign of the nine misses.

| reading | band |
|---|---|
| RESCUED | `kbar`'s median relative error is below `r`'s **and** the misses are no longer >= 7/9 in one direction |
| BETTER BUT STILL BIASED | error falls but the sign bias survives |
| NO BETTER | `kbar`'s median relative error is not below `r`'s |

Under RESCUED the manuscript's diagnosis changes from "the wrong quantity" to "the right quantity,
badly estimated", the Appendix~C sentence is rewritten, and the result is reported explicitly as
post-hoc. Under NO BETTER the existing sentence gains the evidence it currently lacks.

**E3 -- ranking.** Spearman of the nine pairs' `median kbar` against their measured onsets, exact
permutation p. The onset ratio itself ranks at -0.958 against context; a rate that is the *right*
quantity should rank positively against the onset it is supposed to set. Reported with its p, no
band: nine points cannot separate 0.4 from 0.7.

**E4 -- the quantile, as a secondary.** `analysis/onset_theory.py` located the population onset at
the 25th percentile of the `r(x)` distribution to within 0.01 nats on the two pairs it had. Report
the quantile of the `kbar(x)` distribution at which each pair's measured onset sits. Secondary and
descriptive; nine pairs cannot fit a quantile.

## Excluded alternatives (named now so they cannot be adopted afterwards)

1. Dropping a pair because its memoriser is weak. The weak memoriser is exactly the regime where
   the two estimators differ, so excluding it would remove the test.
2. Switching from the median to a quantile chosen after seeing which one lines up.
3. Measuring on a different passage set, split, or truncation than the onset used.
4. Warping either model's logits (`--temperature != 1`) to move the rate.
5. Reporting E2 without E1, or either without the post-hoc disclosure above.
6. Re-describing the original Eq. (req) refutation as anything other than refuted.

## Scoring

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<free card> HF_HUB_OFFLINE=1 \
  HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/exact_rate.py --out results
```

Writes `results/exact_rate.csv` (one row per pair) and `results/exact_rate_per_work.csv`.

---

## Scoring log (appended after the run; nothing above this line is edited)
