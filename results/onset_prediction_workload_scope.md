# Pre-registration: is the reversal's failure the WORKLOAD, or our new pipeline? (feat-170)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why

feat-168 measured, on AlpacaEval-805 with a budget calibrated to bind as hard as on our own corpus,
that the metered decoder **beats** selection: paired `-0.0339 [-0.0534, -0.0137]`. That reading is
now in the manuscript, scoping the headline (`app:workload`), and it rests on **one benchmark
measured once with a pipeline the committed pass did not use.** Two things are wrong with leaving
it there.

**First, the comparison is not like-for-like.** The committed pass is `500` prompts at
`--batch-size 8` and `k=10`; feat-168 is `805` prompts at `--batch-size 64` and `k=1.0`. Batch size
is part of the seed and, at a rate-valued quantity, a shift rather than a re-roll (cautions (u),
(v)). So a reviewer asking *"is that the workload or your new pipeline?"* has no answer in our data,
and neither do we.

**Second, the reading is marginal by this project's own rule.** `-0.0339` sits **`1.71`** interval
half-widths from zero. Caution (ap) fixes what that means from three measured cases: at `2.12` and
`2.43` half-widths a disjoint draw moved a paired difference by `0.0000` and `0.013`, and at
**`1.71`** it moved by `0.061` --- enough to cross zero from here. We are not entitled to lean on
this reading until it is re-drawn, and saying so before re-drawing it is the only way the re-draw
counts.

## What runs

**Arm A --- the workload control.** Our own `850` ordinary prompts (neutral `200`, creative `150`,
factual `500`; the same corpus the committed pass draws its `500` from) through **feat-168's exact
pipeline**: anchor draws at `--trajectories-per-prompt 64`, `--batch-size 64`, the metered decoder
at `k=10` --- which is the matched budget on this corpus by construction, since `8.376%` is
*measured on it* --- the unconstrained opponent, one reward pass, and judge B under order
averaging. Sharded by prompt class across three cards.

**Arm B --- the replication.** AlpacaEval-805 again, same pipeline as feat-168, **`--seeds 52`**
against feat-168's default: a disjoint draw of the same `805` prompts. This is the re-draw
caution (ap) requires at `1.71` half-widths, and it is registered before Arm A is read so neither
can be tuned to the other.

Arm B deliberately carries **no bit-identity gate**. It is an independent draw by construction and
such a gate would be incoherent rather than merely wrong --- the lesson feat-131 recorded.

## Gates, read in this order

- **G0 (both arms' budgets bind).** Each metered cell's activity within `2x` of `0.08376`, and
  under `10%` of its completions byte-identical to the unconstrained opponent. This is feat-168's
  repaired gate and it applies to both arms.
- **G1 (Arm A is the same corpus).** Arm A's prompt ids must be a superset of the committed pass's
  `500`, and its class counts must read `200`/`150`/`500`. A workload control that silently changed
  the workload is worth nothing.
- **G2 (Arm B is a disjoint draw of the same prompts).** Arm B's prompt ids must match feat-168's
  exactly, and its rank-`0` completions must differ from feat-168's on more than `50%` of prompts.
  Identical text would mean the seed did not take.

## Bands

- **B1 --- Arm A, the workload control.** Judge B's paired `D3` on our corpus at feat-168's
  pipeline. **REVERSAL HOLDS** if `> 0` with the interval excluding zero: the failure on AlpacaEval
  is then attributable to the workload, and `app:workload`'s claim stands as written.
  **PIPELINE** if `<= 0` with the interval excluding zero: the failure is our pipeline, not the
  workload, `app:workload` is **withdrawn**, and the committed `+0.0675` itself comes into
  question --- which would be the most consequential finding this project has produced and is
  reported as such. **UNRESOLVED** if the interval contains zero.
- **B2 --- Arm B, the replication.** **REPLICATES** if the sign is negative with the interval
  excluding zero. **DOES NOT REPLICATE** if the interval contains zero or the sign flips. Reported
  with the distance moved, beside caution (ap)'s three cases (`0.0000` at `2.12`, `0.013` at
  `2.43`, `0.061` at `1.71`), so the fourth data point lands in the same table whatever it says.
- **B3 --- what the manuscript then says**, fixed per branch now:
  | B1 | B2 | the appendix |
  |---|---|---|
  | REVERSAL HOLDS | REPLICATES | stands, and gains the control and the replication |
  | REVERSAL HOLDS | DOES NOT REPLICATE | keeps the scope, drops the number, says the size is not established |
  | PIPELINE | either | **withdrawn**, and the pipeline difference is chased before anything else |
  | UNRESOLVED | either | the scope clause stays, the appendix says one benchmark once and unreplicated |

## Excluded in advance

- Pooling Arm B with feat-168, or reading either arm's numbers into the other's band.
- Reading B1 or B2 if G0 fails on that arm.
- Treating a `PIPELINE` reading as a reason to revert to the old pipeline and keep the old number.
  If the pipeline moves this comparison, that is a fact about every number the pipeline produced.
- Quoting a judged level from either arm beside a level from any other pass (caution (ap)).

## What we predict

**B1: REVERSAL HOLDS.** The support-ceiling mechanism predicts the failure is specific to a
workload the `1.8`B anchor cannot do, and our corpus is the one it was chosen for. **B2:
REPLICATES, but we are genuinely unsure** --- at `1.71` half-widths this project's own precedent
says a fresh draw can move a difference by `0.061`, which is `1.8x` the reading itself. If B2 says
DOES NOT REPLICATE we will have caught, by our own rule applied in advance, a number we had
already put in the paper.

## Compute

Host B. Arm A about `1`h`45` sharded over three cards; Arm B about `2`h`50` on one; metered and
opponent cells minutes each; rewards about `15` minutes per arm; judge B about an hour per arm.

## Scoring log
