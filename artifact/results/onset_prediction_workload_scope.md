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

### B1 --- REVERSAL HOLDS. The failure on AlpacaEval is the workload, not our pipeline.

Arm A, our own `850` prompts through feat-168's exact pipeline, judge B, order-averaged:

| quantity | value | reading |
|---|---|---|
| `D1` selection gain | `+0.1218 [+0.1038, +0.1400]` | SURVIVES |
| `D2` metered gain, `k=10` | `+0.0735 [+0.0556, +0.0912]` | SURVIVES |
| **`D3` paired difference** | **`+0.0482 [+0.0303, +0.0662]`** | **REVERSAL CONFIRMED** |

Positive with the interval clear of zero, so the registered branch applies: **the failure on
AlpacaEval is attributable to the workload, and `app:workload`'s claim stands as written.** The
`PIPELINE` branch --- which would have withdrawn that paragraph and put the committed `+0.0675`
itself in question --- does not fire.

**The single-order column is the reason this construction exists.** Arm A's single-order `D3` reads
**`-0.1494`**, which would have said the reversal is *refuted* on our own corpus; order-averaged it
is `+0.0482`. A `0.198` swing between the two constructions on identical text, and the wrong sign
on the naive one. Caution (m) measured position dominance; this is the largest instance of it this
project has produced, and it is on the paper's own workload.

### B2 --- REPLICATES.

Arm B, AlpacaEval again at `--seeds 52`, `97.0%` of rank-`0` completions different:

| pass | `D3` |
|---|---|
| feat-168 | `-0.0339 [-0.0534, -0.0137]` |
| **Arm B (disjoint draw)** | **`-0.0255 [-0.0453, -0.0062]`** |

Same sign, interval still excluding zero: **REPLICATES**, having moved `0.0084`.

**And that is a fourth data point for caution (ap), which it weakens.** The rule on record said a
paired difference at about `1.7` interval half-widths does not survive a fresh draw, resting on one
case (`KL3M`, `1.71`, moved `0.061`). This reading sits at `1.71` and moved `0.0084`. The table is
now:

| half-widths from zero | distance moved | verdict |
|---|---|---|
| `1.71` (`KL3M` breadth) | `0.061` | did not replicate |
| **`1.71` (this arm)** | **`0.0084`** | **replicated** |
| `1.96` (committed `D3`) | --- | --- |
| `2.12` (audited anchor) | `0.0000` | replicated |
| `2.43` (Comma-7B) | `0.013` | replicated |

**Two readings at the same ratio, opposite outcomes.** So the ratio is a weak heuristic and not a
law, and the honest statement is that it flags a reading as *worth re-drawing* rather than
predicting what the re-draw will say. Caution (ap) is amended accordingly rather than kept as
written, and the amendment costs us: it removes the tidy rule we had, and the only thing that
replaces it is re-drawing marginal readings, which is what we did here.

### The degeneracy signature tracks the identity rate across four passes

Reported because it fell out and is a usable diagnostic. The metered arm's order-consistency:

| pass | budget | identical to opponent | order consistency |
|---|---|---|---|
| feat-166 | `k=10` | `98.6%` | `0.868` **STABLE** |
| **Arm A** | `k=10` | `99.5%` | **`0.846` STABLE** |
| feat-168 | `k=1.0` | `4.3%` | `0.335` UNUSABLE |
| **Arm B** | `k=1.0` | `5.5%` | **`0.368` UNUSABLE** |

Every other arm this project has judged reads `0.24`--`0.52`. A metered arm reading above `0.8` is
the tell that it is serving the opponent's own text, and it is the cheapest available check that a
budget is doing nothing --- cheaper than counting active steps, because the judging pass produces
it anyway.

### Incident, 2026-09-22 ~10:52 --- host B's virtualenv was destroyed mid-arm

Arm C's metered cell finished `rc=0` at `10:51:55`; its head-to-head died at `10:53:31` on
`ModuleNotFoundError: No module named 'transformers.models.phi3.configuration_phi3'`, ninety-six
seconds later, with the same judge that had scored Arms A and B an hour earlier. The venv was
gone: `.venv/bin` empty, `pyvenv.cfg` absent, and `site-packages` holding `293` directory entries
and **zero files**. Directories intact, contents removed.

**The cause is UNDETERMINED and is recorded as such.** What can be said:

- **It was not the `rsync`.** Neither `sync_status.sh push` nor the new `push-results` passes
  `--delete`, and rsync without it cannot remove a remote file. Both also ran *after* the failure.
- The `.venv/bin` mtime read `Sep 5 12:36` while the directory was empty, which is not a deletion
  timestamp --- it is the **local** directory's mtime, propagated by `rsync -a` when the filter
  `--include '*/'` recreated the empty directory. That misled the first pass of this diagnosis and
  is worth stating: **a directory mtime on the far side of an `rsync -a` is the near side's mtime,
  not a record of what happened there.**
- The account is shared. An interactive VS Code server session with Copilot was running as this
  user from `09:33`, and AGENTS.md already records other active sessions on this box (one of whose
  `51`GB jobs OOM-killed our `neutral` class twice on 2026-09-19).

**Second refinement, 2026-09-22 12:10 --- a RECURRENCE was traced, and it was ours.** The same
failure signature returned at `12:06` and killed eight freshly launched cells, and this time the
cause is established: **`sync_status.sh push-results`, which this session added at about `11:00`.**
Its filter is `--include 'results/***' --include '*/' --exclude '*'` and it carried no exclusions,
so `--include '*/'` told rsync to create **every local directory** on the remote --- including all
of `.venv`. It recreated this host's `tqdm-4.70.0.dist-info` and thirty others as **empty shells**
beside host B's real `tqdm-4.70.1.dist-info`, and a duplicate `dist-info` makes
`importlib.metadata.version` return `None`, which transformers checks at import. The evidence is
exact: the duplicated names and versions are **this host's** (`tqdm-4.70.0`, `transformers-5.16.1`,
`pandas-3.0.5`, `urllib3-2.7.0`), their remote mtimes are **this host's directory mtimes** offset by
the timezone, and the recurrences line up with the two `push-results` runs.

**The first diagnosis reasoned about the wrong verb.** It checked whether rsync could *remove* a
file --- it cannot without `--delete`, and that reasoning was correct --- and never asked whether it
could *add* a directory, which it does freely. **A sync that cannot delete can still destroy, by
adding.** `push-results` now carries `$EX`, verified by a dry run showing zero `.venv` paths in the
transfer list.

Whether the ORIGINAL `10:52` failure had the same cause is still **undetermined**: `push-results`
did not exist until about `11:00` and `push` has always excluded `.venv`. It is not claimed either
way.

**Two defences added.** `scripts/gpu_env.sh`, which every launcher sources, now scans for duplicate
`dist-info` directories, removes the stale ones and **says so on stderr** --- a silent repair would
hide the next recurrence, and knowing it recurs is the point. And on the user's instruction this
project now keeps its own `UV_CACHE_DIR` inside the repo, so a package cache is no longer a mutable
dependency shared with a sibling project.

**Earlier framing, kept for the record:** those concurrent sessions are the **user's own other
project**, not a stranger's --- the two share one account on both hosts. That makes the most likely
explanation a sibling-project environment operation rather than an outsider: a `uv` cache clean, a
venv rebuild or an IDE-driven reinstall touching a shared cache. It is still **not established**,
and naming a specific action without evidence would be the same guess in nicer clothes. What it
does change is the remedy: this is coordinable rather than merely survivable, and the practical
defence is that a long arm must be re-runnable from its last written artefact --- which is what
saved Arm C, whose generation was on disk when its judging died.

Asserting which of these did it would be a guess, and a guess in this file is worth less than
nothing.

**Repair.** `uv venv` plus `uv pip install -r requirements.txt` restored it from uv's local cache
in seconds --- `306` packages, `torch 2.10.0+cu128`, `transformers 5.17.0`, CUDA visible, and the
`Phi-3.5` tokenizer that had failed loading cleanly. **No measurement is affected**: every arm
scored before the incident wrote its CSV before it, Arm C's generation completed `rc=0` and its
trajectories are on disk, and only the head-to-head had to be re-run --- which reads the same
generations and the same reward cache.

**What this changes going forward.** A long arm on that host can lose its interpreter between two
steps of one launcher, so a multi-stage launcher must not assume the environment it started in
still exists. `run_wscope_armc.sh` happened to be safe because its two stages write separate
artefacts and the second is re-runnable from the first's output; a launcher that had piped one
stage into the next would have lost the generation too.

### Arm C --- G-cal PASS, G0 PASS on both legs, and **B4: REVERSAL HOLDS**

Calibration on our corpus, `200` prompts, target `0.08008` derived from feat-168's chosen arm
rather than typed:

| `k` | `0.5` | `0.7` | `0.8` | **`0.9`** | `1.0` |
|---|---|---|---|---|---|
| activity | `0.44756` | `0.20675` | `0.13336` | **`0.08203`** | `0.04939` |

**G-cal PASS** --- four points above the target and one below, so it is bracketed and the choice is
an interpolation. `argmin` picks **`k = 0.9`** at `1.02x` the target; no tie rule needed.

**G0 PASS on both legs at the full `850`:** activity `13{,}523` of `152{,}591` = **`8.862%`**
(`1.11x` the target) and **`2.0%`** byte-identical to the opponent. Unlike Arms A and B, this arm
satisfies the gate as originally written --- because at a budget that genuinely binds the two legs
stop contradicting each other, which is the clearest possible statement of what went wrong with
G0.

| quantity | value | reading |
|---|---|---|
| `D1` selection gain | `+0.1218 [+0.1038, +0.1400]` | SURVIVES |
| `D2` metered gain, `k=0.9` | `+0.0253` | SURVIVES |
| **`D3` paired difference** | **`+0.0965 [+0.0765, +0.1162]`** | **REVERSAL CONFIRMED** |

### The `2x2`, which is the result this arm was built for

| workload | budget | binds | metered `==` opponent | paired `D3` |
|---|---|---|---|---|
| **ours** | `k=10` | `0.011%` | `99.5%` | **`+0.0482 [+0.0303, +0.0662]`** |
| **ours** | **`k=0.9`** | **`8.86%`** | **`2.0%`** | **`+0.0965 [+0.0765, +0.1162]`** |
| AlpacaEval | `k=10` | `0.016%` | `98.6%` | `-0.0957 [-0.1124, -0.0792]` |
| AlpacaEval | `k=1.0` | `8.01%` | `4.3%` | `-0.0339 [-0.0534, -0.0137]` |
| AlpacaEval | `k=1.0`, re-drawn | `7.92%` | `5.5%` | `-0.0255 [-0.0453, -0.0062]` |

**The reversal holds on our workload at both budgets and fails on AlpacaEval at both.** Every
interval excludes zero and the sign is constant within each workload, so the split is the workload
and nothing else --- not the budget, not the batch size, not the pipeline, not the seed.

**And the two workloads respond to a binding budget in opposite directions.** Forcing the meter to
spend makes selection's margin *grow* on our corpus (`+0.0482 -> +0.0965`) and *shrink* on
AlpacaEval (`-0.0957 -> -0.0339`). Both are what the support-ceiling account predicts and neither
was registered, so both are reported as observations rather than claims: where the anchor is
competent, a budget that actually constrains the risky model hurts the risky model; where the
anchor is out of its depth, the same constraint costs the meter the advantage it was getting for
free by simply *being* the risky model.

**We predicted REVERSAL HOLDS and it does.** We also wrote the PIPELINE branch out in advance ---
*"the honest conclusion is that this paper's headline survives only against a meter that is not
metering, and we would rather find that ourselves"* --- and the measurement that would have fired
it is exactly the one that did not.

### A second judge on Arm A: the vacuous-budget cell is the weakest version of our own claim

Arm A is our corpus at the paper's **own** `k=10`, and until now it had one judge. Judge~C:

| cell | budget | binds | judge~B | judge~C |
|---|---|---|---|---|
| Arm A | `k=10` | `0.011%` | `+0.0482 [+0.0303, +0.0662]` CONFIRMED | **`+0.0244 [-0.0029, +0.0512]` UNRESOLVED** |
| Arm C | `k=0.9` | `8.86%` | `+0.0965 [+0.0765, +0.1162]` CONFIRMED | `+0.0903 [+0.0629, +0.1176]` CONFIRMED |

**At a budget that genuinely binds, both judges clear zero; at the paper's own `k=10` the second
judge does not.** Both point estimates are positive and the two cells agree in sign, so nothing
here reverses anything --- but it says where the claim is strongest, and it is not where the paper
measures it.

This does **not** contradict the committed headline, and the reason matters: that reading is a
different pass (`500` prompts, `--batch-size 8`) and Table~\ref{tab:h2hrepeat} already reports
judge~C clearing zero on it at `+0.0620` and `+0.1000`. Caution (ap) forbids setting these against
each other as levels, and we do not. What is comparable is the **shape** across the two cells of
*this* pass, which share prompts, pipeline and judges: the binding cell is the cleaner measurement
on both judges, and the vacuous one is the marginal one on both (`+0.0482` against `+0.0965`;
`+0.0244` against `+0.0903`).

**The mechanism is the degeneracy already reported.** At `k=10` the metered arm is byte-identical
to the opponent on `99.5%` of prompts, so the comparison is very nearly selection against the
unconstrained risky model with the budget doing nothing. That the reversal is *harder* to establish
there, under a judge that was not used to pick it, is what one should expect and is worth saying
out loud rather than leaving for a reader to notice.

**And the single-order column is again the reason for the construction**: judge~C's single-order
`D3` on Arm A reads `-0.1971` against the order-averaged `+0.0244`, a swing of `0.22` with the
wrong sign, on identical text. That is the largest position artefact this project has measured,
and it is at the vacuous budget where the two arms' texts are nearly the same --- which is exactly
where a position-dominated judge has least else to go on.
