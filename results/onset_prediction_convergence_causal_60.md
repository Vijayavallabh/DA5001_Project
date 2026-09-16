# Retry at 60 epochs: is non-convergence the *cause* of the irreproducible onset ratio?

**Committed 2026-09-16 11:35, before any of these memorisers exists.** Nothing above the
`## Scoring log` heading is edited after that line is written.

## Why this arm exists

`results/onset_prediction_convergence_causal.md` probed `lr` 1.5e-4, 1e-4 and 5e-5 at the recipe's
own 40-epoch cap and **none crossed the `0.02` stop-loss** (`0.0229`, `0.0260`, `0.0305`), so that
stage is INVALID by its own written condition and its stage 2 was not run.

The rates worked; the cap did not allow them to finish. At `lr 1.5e-4` the final four epochs read
`0.0269`, `0.0254`, `0.0236`, `0.0229` --- monotone and still descending --- against the published
`lr 3e-4`'s non-monotone `0.0623 / 0.0865 / 0.0258 / 0.1098`. The oscillation the intervention
targeted is gone; only the threshold was not reached.

**No ratio has been measured.** No sweep has run on any of those memorisers and no onset exists.
This re-attempts a construction that failed to construct, which is a different act from re-running
an experiment whose answer one dislikes; the record above states that plainly, and had a single seed
been swept the honest course would have been to stop instead.

## The single change

`--epochs 60` in place of `40`. Everything else is carried over verbatim: the three rates, the
selection rule, the bands, the entry gate, both invalidity conditions, the corpus, the anchor, the
grid and the passages.

> **The rate carried into stage 2 is the LARGEST probed rate whose stop-loss fires within 60
> epochs.** Unchanged from the original arm.

## Committed bands, unchanged

The quantity is the **three-seed ratio span** at the chosen rate, read against the same cell's
three-seed span at the published `lr 3e-4`, which is `0.2597`.

| outcome | reading |
|---|---|
| span **`< 0.10`** | **convergence is the cause.** The ratio becomes reproducible once the fine-tune converges, on the cell where it did not, with pair, corpus, anchor, grid and passages fixed |
| span **`>= 0.20`** | **convergence is not the cause.** The explanation now in `appendix_robustness.tex` and `appendix_limitations.tex` is withdrawn |
| anything else | **inconclusive**, reported as inconclusive |

**Entry gate, unchanged:** a point enters only if its sampled `k=-1` is at least `0.10`.

**The diagnostic and the confound, unchanged.** The strength band is reported either way. If the
converged ladder's sampled `k=-1` lands far outside the `0.150`–`0.239` the published-rate ladder
occupied, the intervention moved convergence **and** strength, the two are not separated, and that
is disclosed as a confound rather than reported as a clean result.

**A second, longer training run is itself a change**, and it is one the paper must state: 60 epochs
is not the published recipe either. What this arm can establish is that a *converged* memoriser on
this cell reproduces its onset ratio; it cannot establish that the published recipe does, because
the published recipe does not converge here. That limit is written down now, not discovered later.

## What makes this arm INVALID rather than a failure

- **No probed rate converges within 60 epochs.** We failed to build the intervention again.
  INVALID, and **the arm is then abandoned, not re-probed a third time** --- a construction that
  fails twice under a doubled budget is telling us the cell does not converge under this recipe
  family, which is itself the finding and is reported as one.
- **The converged memoriser falls below the entry gate of `0.10`.** INVALID.

## What will not happen

No third epoch cap, no fourth rate, no re-grid, no re-threshold, no fourth seed, no dropping a seed,
and no stage 2 if stage 1 leaves the arm invalid.

## Scoring log
