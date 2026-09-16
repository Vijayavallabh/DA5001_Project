# The other half of the crossover: break a converged cell and see if reproducibility goes with it

**Committed 2026-09-16 10:50, before any of these memorisers exists.** Nothing above the
`## Scoring log` heading is edited after that line is written.

## Why one direction is not enough

`results/onset_prediction_convergence_causal.md` takes the one cell whose fine-tune never converges
and makes it converge. If its ratio becomes reproducible, that is one arrow. It is still consistent
with convergence being a *marker* of something else — this pair, this corpus, this memoriser's
particular weakness — that the lower learning rate also happened to fix.

This is the other arrow, on a different pair. **KL3M-520M on BookMIA converges tightly and
identically at every seed** — 26 of 40 epochs at final losses `0.0169`, `0.0173`, `0.0177` against a
`0.02` stop-loss — and its three-seed ratio span is `0.0663`, one of the three that reproduce. Raise
its learning rate until the stop-loss stops firing and nothing changes but the optimiser's
stability.

Together the two arms are a crossover: convergence is made and unmade on two different pairs, in
opposite directions, with everything else held.

## Design

**Stage 1, the probe.** Three fine-tunes at `--lr 6e-4`, `1e-3`, `2e-3`, seed 0, everything else
identical to the cell's own recipe (`--epochs 40 --rank 128 --batch 2 --accum 4 --stop-loss 0.02
--target-modules all-linear --no-chat --max-len 0`, base `output/phase5/anchor_kl3m-002-520m`,
corpus `bookmia100_onset600.jsonl`).

> **The rate carried into stage 2 is the SMALLEST probed rate whose stop-loss does NOT fire within
> 40 epochs.** Committed now. Smallest, for the same reason the forward arm takes the largest: the
> intervention should be the least departure from the published recipe that achieves the effect.

**Stage 2, the ladder.** Seeds `1` and `2` at that rate, joining the probe's seed 0, swept on this
cell's own committed grid — `-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2`, **without** the
extension Pleias needed, since KL3M's no-crossing there was `0.0%` — `--modes single --limit 100`.

## Committed bands

The quantity is the **three-seed ratio span**, read against this same cell's converged three-seed
span of `0.0663`.

| outcome | reading |
|---|---|
| span **`>= 0.20`** | **convergence is causal in both directions.** Breaking it on a cell that reproduced destroys the reproduction, as making it on a cell that did not restores it. The paper states convergence as a cause |
| span **`< 0.10`** | **convergence is not sufficient.** A non-converged memoriser on this pair still reproduces, so convergence cannot carry the explanation on its own and the forward arm's result, whatever it is, must be stated as specific to that cell |
| anything else | **inconclusive**, reported as inconclusive. No fourth seed, no second rate |

**Entry gate, unchanged.** A point enters only if its sampled `k=-1` recall is at least `0.10`.

## The confound this arm shares with its partner, stated before the numbers exist

Raising the learning rate will plausibly make the memoriser **weaker** as well as unstable, and
lowering it in the forward arm may make it **stronger**. Strength is the variable the onset ratio
tracks (`rho = -0.714` over the seven pooled points), so both arms risk moving convergence and
strength together — which is the very confound the crossover exists to break.

The reported diagnostic is therefore the **strength band**, and the reading is committed now:

- If the broken ladder's sampled `k=-1` stays inside the converged ladder's `0.669`–`0.733`, then
  convergence moved and strength did not, and the arm is clean.
- If it lands far outside, the arm is **confounded and is reported as confounded** — the span is
  still given, and it is not claimed as evidence that convergence is the cause.
- If it falls below the entry gate of `0.10`, the intervention destroyed the memorisation rather
  than its stability: **INVALID**, on the same principle as the forward arm.

## What makes this arm INVALID rather than a failure

- **No probed rate breaks convergence.** We failed to build the intervention. INVALID, and not
  evidence that convergence is irrelevant.
- **The broken memoriser falls below the entry gate.** INVALID, as above.

## What will not happen

No second rate sweep after seeing a ratio, no fourth seed, no re-grid, no re-threshold, no dropping
a seed, and no borrowing the extension points Pleias' grid carries and this cell's does not.

## Scoring log
