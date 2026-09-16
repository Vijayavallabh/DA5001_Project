# Reverse arm, second attempt: rates between converging and destroying

**Committed 2026-09-16 13:10, before any of these memorisers exists.** Nothing above the
`## Scoring log` heading is edited after that line is written.

## Why this arm exists

`results/onset_prediction_convergence_reverse.md` probed `6e-4`, `1e-3`, `2e-3` to find the smallest
rate that stops KL3M-520M's stop-loss from firing on BookMIA. The three rates bracket the transition
but do not sample it:

| rate | epochs | final loss | outcome |
|---|---|---|---|
| `6e-4` | 25/40 | `0.0189` | **converges** — sampled `k=-1` `0.855`, a healthy memoriser |
| `1e-3` | 40/40 | `4.4826` | **destroyed** — final loss above its own epoch-1 loss (~2.8) |
| `2e-3` | 40/40 | `3.7572` | **destroyed** |

By the committed rule the carried rate is the smallest that does not converge, which is `1e-3`. Its
loss of `4.4826` is not an unstable memoriser but a wrecked model, and the committed invalidity
condition — *"the broken memoriser falls below the entry gate of `0.10`: INVALID"* — is being
measured directly (`output/phase5/revgate_lr1E3`, a `k=-1`-only probe at the cell's own protocol,
byte-identical `[ca]` line).

The interval `6e-4`–`1e-3` is where a rate that **destabilises without destroying** must live, if one
exists. This arm samples it at `7e-4`, `8e-4`, `9e-4`.

**No onset ratio has been measured, on this arm or any of its memorisers.** No sweep over a budget
grid has run; the only decode is the entry-gate probe, which is the *admissibility check* the
protocol requires before a point may enter at all, not the quantity under test. Re-attempting a
construction that failed to construct is legitimate on exactly that basis, as it was for the forward
arm; had a single ratio been computed the honest course would be to stop.

## The single change

Three additional rates. The candidate set becomes `{6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 2e-3}` and the
rule is unchanged:

> **The rate carried into stage 2 is the SMALLEST probed rate whose stop-loss does NOT fire within
> 40 epochs, and whose memoriser clears the `0.10` entry gate.** The gate clause was already an
> invalidity condition in the first arm; stating it inside the selection rule makes explicit what
> that arm left implicit, and it cannot favour any outcome because both halves were committed there.

Everything else is carried over verbatim: `--epochs 40 --rank 128 --batch 2 --accum 4 --stop-loss
0.02 --target-modules all-linear --no-chat --max-len 0`, base `output/phase5/anchor_kl3m-002-520m`,
corpus `bookmia100_onset600.jsonl`, and the cell's own grid without Pleias' extension.

## Committed bands, unchanged

The three-seed ratio span at the chosen rate, against this cell's converged span of `0.0663`.

| outcome | reading |
|---|---|
| span **`>= 0.20`** | breaking convergence destroys reproduction on a cell that had it |
| span **`< 0.10`** | convergence is not sufficient; a non-converged memoriser here still reproduces |
| anything else | **inconclusive**, reported as inconclusive |

**The confound, unchanged and now sharper.** A rate that destabilises will also weaken the
memoriser, and strength is what the ratio tracks. The reading committed in the first arm stands: the
arm is clean only if the broken ladder's sampled `k=-1` stays inside the converged ladder's
`0.669`–`0.733`; outside that it is **reported as confounded**, and below `0.10` it is **INVALID**.
With `6e-4` sitting at `0.855` and `1e-3` in ruins, a rate in between may well land in neither the
clean band nor the invalid one, and that middle outcome is reported as confounded rather than
quietly used.

## What makes this arm INVALID, and when it is abandoned

- **No probed rate both breaks convergence and clears the entry gate.** INVALID, and **the arm is
  then abandoned, not probed a third time.** The forward direction of this crossover was abandoned
  on exactly this rule after two attempts; the same standard applies here, and the finding — that on
  this pair the optimiser goes from converging to destroying with no usable band between — is
  reported as the result of having looked.

## What will not happen

No fourth set of rates, no change to the epoch cap, no re-grid, no re-threshold, no fourth seed, no
dropping a seed, and no stage 2 if stage 1 leaves the arm invalid.

## Scoring log
