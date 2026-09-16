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

## Scoring, 2026-09-16 --- INVALID and ABANDONED. The stop-loss is below this pair's floor.

```
bash scripts/run_convergence_probe.sh <lr> <gpu> 60     # 11:31-12:46, GPUs 0/1/4
```

| probe | epochs | final loss | stop-loss | converged? |
|---|---|---|---|---|
| `lr 1.5e-4` | 60/60 | `0.0222` | 0.02 | no |
| `lr 1e-4` | 60/60 | `0.0222` | 0.02 | no |
| `lr 5e-5` | 60/60 | `0.0284` | 0.02 | no |

**No rate converged under a doubled epoch budget, so by the condition written down before the run
this arm is INVALID and is now ABANDONED.** No third probe, exactly as committed. Stage 2 never ran,
no sweep exists, and no onset ratio was ever computed from any of these six memorisers.

### The construction did not fail for the reason the retry assumed, and that is the finding

The 40-epoch stage read `0.0229` at `lr 1.5e-4`, monotone and still descending, so the retry assumed
the epoch cap was binding. It was not. At 60 epochs the last three epochs read:

| rate | epoch 58 | epoch 59 | epoch 60 |
|---|---|---|---|
| `1.5e-4` | `0.0222` | `0.0222` | `0.0222` |
| `1e-4` | `0.0222` | `0.0223` | `0.0222` |

**Two different learning rates, two different optimisation trajectories, the same value to four
decimals, flat for three epochs.** That is a floor. Pleias-1.2B at rank 128 cannot drive its loss on
these 600 BookMIA excerpts below about `0.0222`, and the recipe's stop-loss of `0.02` sits
underneath it.

So *"this cell never converges"* --- which the paper says of it, and which this whole crossover was
built to exploit --- is **not a statement about the optimiser**. It is a threshold set below what the
pair can reach. Two distinct things had been collapsed into one phrase:

1. At the published `lr 3e-4` the run is genuinely **unstable**: `0.0623 / 0.0865 / 0.0258 / 0.1098`
   across epoch counts, `0.0619`–`0.1143` across seeds. Lowering the rate removes this completely.
2. Underneath that instability there is a **floor at `0.0222`**, above the `0.02` threshold, which no
   rate in `5e-5`–`3e-4` and no budget up to 60 epochs gets past.

KL3M-520M on the same corpus reaches `0.0169` at 25--26 epochs without difficulty. The difference
between the two cells is not that one optimiser misbehaved; it is that one pair can memorise this
corpus to below `0.02` and the other cannot.

### What this does to the paper, and what it does not

The convergence column of Table~2 and the sentences built on it stay factually correct --- those
memorisers did not reach their stop-loss --- but the *reading* offered for them was too narrow, and
is corrected in `appendix_robustness.tex` and `appendix_limitations.tex`: a memoriser that never
fires its stop-loss may be unstable, or may simply be a pair whose achievable loss lies above the
threshold, and on the one cell measured here it is both.

**The forward direction of the crossover cannot be built at all**, because the intervention it needs
--- a converged memoriser on this cell --- does not exist under this recipe family. That is reported
as the result of trying, not hidden as an abandoned branch. The reverse direction is unaffected and
is scored separately in `results/onset_prediction_convergence_reverse.md`.

**Cost of the attempt:** six fine-tunes, about 4.3 GPU-hours, no sweeps.
