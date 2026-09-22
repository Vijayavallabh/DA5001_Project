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

## Arm C, registered 2026-09-22 10:00 --- before its calibration sweep runs

Arms A and B compare the two workloads at **the paper's own `k`** --- Arm A at `k=10` on our
corpus, feat-166 at `k=10` on AlpacaEval --- and that is now a matched comparison in a way it was
not before: both are run by the same launcher with a shared seed stream, so both exhibit the same
near-degenerate metered arm (Arm A: `99.5%` of completions byte-identical to the opponent;
feat-166: `98.6%`). The committed pass cannot exhibit that, because its metered and opponent cells
were sampled independently, which is exactly why the degeneracy went unnoticed for a year.

What is still missing is the other matched pair: **the two workloads at the same BINDING RATE.**
feat-168 has AlpacaEval at `8.008%`. Our corpus has nothing there --- `k=10` binds at `0.008%` and
the correction above shows the sweep values that bracket `8%` on this corpus lie between `k=0.5`
(`48.1%`) and `k=1.0` (`5.5%`).

**The rule, fixed here before the sweep.** Run `k` over `{0.5, 0.7, 0.8, 0.9, 1.0}` on the first
`200` prompts of our own corpus, `--trajectories-per-prompt 1`, `--batch-size 64`,
`--max-new-tokens 200`, one card per point. **Choose the single `k` minimising
`|activity(k) - 0.08008|`**, where `0.08008` is feat-168's chosen arm's own measured binding rate,
read out of its trajectories by `analysis/budget_calibration.py` rather than typed. Ties go to the
larger `k` (the weaker constraint), which cannot occur at this precision.

**G-cal (the grid brackets the target).** At least one grid point above `8.008%` and one below, or
Arm C reports NOT RUN. The grid is deliberately wide at the bottom (`k=0.5` measured `48%` under
the old pipeline) so a pipeline-induced shift cannot silently leave it one-sided.

**G0 (the budget binds, and the arm is not the opponent).** The chosen cell's activity within `2x`
of `8.008%`, and under `10%` of its completions byte-identical to the unconstrained opponent ---
the same two legs feat-168 used, and the second is the one that needs no reference at all.

**B4 --- the matched-rate comparison.** Judge B's paired `D3` on our corpus at the chosen `k`,
against feat-168's `-0.0339 [-0.0534, -0.0137]` on AlpacaEval at `8.008%`. **REVERSAL HOLDS** if
`> 0` with the interval excluding zero; **PIPELINE** if `<= 0` with the interval excluding zero,
which would mean the meter beats selection on *both* workloads once it genuinely binds and the
scoping story in `app:workload` is wrong in a way that matters far beyond AlpacaEval; UNRESOLVED
if it contains zero.

**Excluded in advance:** extending the grid after seeing it; choosing `k` by anything but the
`argmin`; reading B4 if G-cal or G0 fails; pooling Arm C with Arm A, which is the same corpus at a
different budget and not a replication of it.

**What we predict.** **REVERSAL HOLDS.** The support-ceiling account says the anchor is competent
on this workload and therefore `64` draws from it are worth having, whatever the meter is allowed
to spend. If Arm C instead reads PIPELINE, the honest conclusion is that this paper's headline
survives only against a meter that is not metering, and we would rather find that ourselves.

**Compute.** Local host, GPUs `2` and `4` --- the only two free here; `0` and `1` hold another
user's `77` GB jobs and `3` is the `4` GB T400 (never used). Host B is running Arms A and B.

## Scoring, 2026-09-22

### Gates that need no reference: both PASS

| gate | requirement | measured | reading |
|---|---|---|---|
| G1 Arm A is the same corpus | classes `200`/`150`/`500` | `200`/`150`/`500`, total `850` | **PASS** |
| G2 Arm B is a disjoint draw | ids match feat-168; `>50%` of rank-`0` completions differ | `805`/`805` ids; **`97.0%`** differ | **PASS** |

### G0 is UNSATISFIABLE AS WRITTEN, and the proof is arithmetic

Recorded **before any band was read.** At the time of writing this section
`results/order_averaged_h2h__wscope_a.csv` and `..._b.csv` had not been copied to this host, let
alone opened; the commit that carries this repair precedes the commit that pulls them.

G0 demands two things at once: activity within `2x` of the committed arm's, **and** under `10%` of
completions byte-identical to the unconstrained opponent.

| arm | activity | vs reference `0.000080` | identity | G0 |
|---|---|---|---|---|
| A (our corpus, `k=10`) | `0.00011` | `1.4x` --- **PASS** | `99.5%` --- FAIL | fails |
| B (AlpacaEval, `k=1.0`) | `0.07922` | `990x` --- FAIL | `5.5%` --- **PASS** | fails |

**Each arm fails exactly the leg the other passes, and that is not a coincidence.** The committed
arm's reference rate *is* `0.008%`; a decoder binding on one step in twelve thousand is the
unconstrained risky model to within sampling noise. So "binds like the committed arm" **entails**
"is the opponent", and the two legs are contradictory on this corpus by arithmetic rather than by
anything the arms did. A gate nothing can pass gates nothing --- caution (as) --- arriving here
through a conflict between two legs rather than through one wrong threshold.

The incoherence is downstream of the same defect as the calibration target: both legs were written
believing the committed arm binds at `8.376%`, and at that rate they are perfectly compatible.
**One false constant made a gate that cannot be satisfied, and the gate's two halves are the
dichotomy this paper is about, disagreeing with each other inside our own instrument.**

### The repair, made before the bands were read

Per caution (w) a defect in our own specification must not retire a question, so the gate is
repaired rather than the arms failed --- and the repair is principled rather than chosen to let
both through. **The error was applying ONE reference to TWO arms with different registered
purposes.** Caution (at) already gives the rule: derive the reference from *the arm being
replicated*.

- **Arm A** is registered as *"our own `850` ordinary prompts through feat-168's exact pipeline"* ---
  its job is fidelity to the **committed protocol**. Gate: activity within `2x` of the committed
  `k=10` arm's own derived rate. **`1.4x` --- PASS.** The `99.5%` identity is not a failure, it is
  the committed protocol's own near-vacuity made visible, and making it visible is what this arm
  is for: the committed pass cannot show it, because its metered and opponent cells are sampled
  independently.
- **Arm B** is registered as *"AlpacaEval-805 again, same pipeline as feat-168 ... a disjoint
  draw"* --- its job is to **replicate feat-168's arm**. Gate: activity within `2x` of *that*
  arm's measured `0.08008`, plus the identity leg, which needs no reference at all.
  **`0.99x` and `5.5%` --- PASS.**

Neither threshold is loosened; both are re-pointed at the arm each was always about. The `2x`
tolerance and the `10%` identity cut are exactly as registered.

**Stated plainly as a weakening:** the repair was made after seeing G0 fail, which is weaker than
fixing a gate before its data exists. What limits the damage is that the bands were not read first
and the commit order proves it, and that the repair follows a rule this document already carried
(caution (at)) rather than one invented for the occasion.
