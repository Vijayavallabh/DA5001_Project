# Pre-registration: a budget that binds on AlpacaEval, so Mixtral's question can be asked (feat-168)

Committed **before the calibration sweep runs**, so the budget cannot be chosen to suit an answer.
Nothing above the `## Scoring log` line is edited afterwards.

## Why

feat-166 ran the registered arm and **G0 failed**: judge B's paired difference read
`-0.0957 [-0.1124, -0.0792]` where the gate required it positive and clear of zero. The cause is
measured, not guessed --- at `k=10` on AlpacaEval the budget is active on `26` of `160,227` steps
(`0.016%`, against `8.376%` for the same budget on our own corpus) and `794` of `805` served
completions are the unconstrained opponent **byte for byte**. The arm labelled "metered decoder"
was the risky model. Per caution (w) that is a defect in our specification, so feat-166 is INVALID
and the question it was built to answer --- whether Mixtral's `+0.0090 [-0.0355, +0.0530]` is a
small effect or a wide interval --- is untouched and still open.

This arm rebuilds the instrument and asks the same question with the same bands.

## The rule that picks the budget, fixed here before any sweep

A budget is not transferable between corpora; what transfers is **how hard it binds**. The
committed pass's metered arm is active on `8.376%` of decode steps (`261,239` of `3,118,893`,
`output/phase2/conc_all`, the three counters that partition the true decoded length per caution
(ah)). So:

> Run `k` over the grid `{0.1, 0.3, 1.0, 3.0, 10.0}` on the first `200` AlpacaEval prompts, one
> card each, `--trajectories-per-prompt 1`, `--max-new-tokens 200`. Compute each cell's activity
> rate the same way. **Choose the single `k` minimising `|activity(k) - 0.08376|`.** Ties, which
> cannot occur at this precision, go to the smaller `k`.

The grid is fixed here, the target is a number already on record, and the selection is `argmin` of
one distance --- there is no free choice left at selection time. **The calibration sweep reports
activity rates only.** No judge runs on it, no utility is computed from it, and no band below is
readable from it; an activity rate is an instrument property, in the same sense that reading a
generation's shape is not peeking (caution (au)).

**G-cal (the grid brackets the target).** At least one grid `k` must sit above `8.376%` activity
and at least one below. If the whole grid is on one side the target is not reachable on this corpus
and this arm reports NOT RUN rather than picking the nearest endpoint --- an endpoint is a ceiling,
not a match, and caution (g) is what a ceiling does to an interval.

## What then runs

The metered cell only, at the chosen `k`, on all `805` prompts, `--batch-size 64` as feat-166 used
so the two cells of this pass stay one pipeline. **The `k=0` anchor draws (`n=64`), the opponent
and the `51,520` reward scores are NOT re-run** --- they do not depend on `k`, they are on disk from
feat-166, and re-drawing them would change two things at once (caution (v)). Then
`analysis/order_averaged_h2h.py` under judge B `Phi-3.5-mini` and `Mixtral-8x7B-Instruct`, exactly
as feat-166 specified.

## Gates, read in this order

- **G0 (the budget now binds).** The chosen cell's activity rate must be within `2x` of `8.376%`
  in either direction, and fewer than `10%` of its completions may be byte-identical to the
  unconstrained opponent. feat-166 read `98.6%`; the committed pass reads `0%`. This is the gate
  feat-166 did not have and it is the reason it failed silently.
- **G1 (the pipeline reproduces itself).** Judge B's paired `D3` must be positive with its interval
  clear of zero, as in feat-166. A direction gate, not a magnitude gate.
- **G2 (the sample is bigger).** Judge B's interval half-width below the `0.0348` on record.

## Bands

Unchanged from feat-166, which is the point --- they were committed before any of this and are not
being rewritten after a failure.

| reading | band |
|---|---|
| **REVERSAL HOLDS** | Mixtral's difference `> 0` and interval excludes zero |
| **TIGHT ZERO** | interval contains zero **and** half-width `< 0.030` |
| **STILL UNRESOLVED** | interval contains zero with half-width `>= 0.030` |
| **INVERTS** | difference `< 0` and interval excludes zero |

B2's four sentences carry over verbatim from feat-166. B3 carries over: both judges' paired `g_sel`
and `g_met`, and Mixtral's order-consistency against the `0.358`--`0.426` on record.

## Excluded in advance

- Choosing `k` by anything but the `argmin` rule above, or extending the grid after seeing it.
- Reading any band if G0 or G1 fails.
- Pooling with feat-166's INVALID pass, or with the committed `500`-prompt pass.
- Quoting the activity rates from the calibration sweep as a result about the workload. The
  measured `0.016%` at `k=10` belongs to feat-166 and is already reported there.

## What we predict

**G0 passes at `k` between `0.3` and `1.0`.** The committed corpus mixes classes where the anchor
and the risky model disagree sharply; AlpacaEval is instruction text where a `1.8`B base anchor and
an instruct model diverge constantly, so the same activity should need a much tighter budget. On
B1 the prediction is unchanged from feat-166: **TIGHT ZERO**, because `+0.0090` sits `0.20`
interval half-widths from zero and caution (ap) says a reading that marginal does not survive a
fresh draw as anything but marginal.

We also state the uncomfortable possibility in advance: if a binding budget makes the metered arm
much weaker, `D3` may read positive for reasons that have nothing to do with the judge, and G1
passing would then be necessary but not sufficient. B1 is a statement about **Mixtral against judge
B on the same text**, and that is how it will be read --- the two judges scoring one pass, never a
level from this pass against a level from another (caution (ap)).

## Compute

Host B. Calibration `5` cards, about `25` minutes. The metered cell about `40` minutes. Judge B
about an hour, Mixtral on two cards about `6` hours.

## Scoring log

### Calibration, 2026-09-22 --- recorded before the metered cell ran

`analysis/budget_calibration.py`, `results/mixpow_kcal.csv`. Target `0.08376`, the committed
metered arm's own activity over the three counters that partition the true decoded length.

| `k` | active steps | total steps | activity | `|dist|` |
|---|---|---|---|---|
| `0.1` | `21,103` | `27,571` | `0.76541` | `0.68165` |
| `0.3` | `21,403` | `31,306` | `0.68367` | `0.59991` |
| **`1.0`** | `2,595` | `33,880` | **`0.07659`** | **`0.00717`** |
| `3.0` | `173` | `37,868` | `0.00457` | `0.07919` |
| `10.0` | `6` | `39,225` | `0.00015` | `0.08361` |

**G-cal PASS** --- `2` grid points above the target and `3` below, so the target is bracketed and
the choice is an interpolation rather than an endpoint. **CHOSEN `k = 1.0`**, activity `0.07659`,
`0.91x` the target. The rule was `argmin |activity(k) - 0.08376|` and `1.0` wins it by a factor of
`11` over the next-nearest grid point, so no tie rule was needed.

This lands inside the `0.3`--`1.0` range the registration predicted. The curve is worth recording
because it is steep in exactly the place the paper's dichotomy says it should be: activity falls
`0.684 -> 0.077 -> 0.005` across one decade of `k` on either side of `1.0`, so on this corpus the
budget goes from binding at two thirds of all steps to binding at one in two hundred over a `30x`
change. There is no wide plateau where a per-token budget is *mildly* active --- which is the
dichotomy's shape, measured on a public benchmark.

The metered cell is now generating at `k = 1.0` on all `805` prompts. The anchor draws, the
opponent and the `51,520` reward scores are feat-166's, unchanged.

## Scoring, 2026-09-22

### Gates, in the registered order

| gate | requirement | measured | reading |
|---|---|---|---|
| G0a budget binds | activity within `2x` of `0.08376` | **`0.08008`** (`11,499` of `143,598`), `0.96x` | **PASS** |
| G0b not the opponent | under `10%` byte-identical | **`4.3%`** (`35`/`805`) | **PASS** |
| G1 direction | judge B's paired `D3 > 0`, interval clear of zero | **`-0.0339 [-0.0534, -0.0137]`** | **FAIL** |
| G2 power | half-width below `0.0348` | `0.0199` | pass |

**G0 passes decisively and that is the whole point of this arm**: at `k = 1.0` the budget is active
on `8.008%` of steps against the committed pass's `8.376%`, and `4.3%` of completions match the
opponent against feat-166's `98.6%`. The calibration also held from `200` prompts to `805`
(`0.07659 -> 0.08008`), which it did not have to.

A third, unregistered confirmation fell out of the judging and is worth recording because it was
the tell that started all of this: feat-166's metered arm had an order-consistency of **`0.868`**,
flagged STABLE, the only arm any pass in this project has produced that a judge could tell apart
reliably. At `k = 1.0` the same arm reads **`0.3354`**, UNUSABLE, in line with every other arm in
every other pass. The anomaly was the arm being the opponent, and repairing the budget removed it.

### G1 FAILED, and this time it is not an instrument defect

The registration excludes *"Reading any band if G0 or G1 fails"*. **No branch of B1 is claimed, the
Mixtral judge was not run, and Section 4's wording stands.** B1 remains open after two attempts.

But the two failures are not the same kind and the difference is the finding. feat-166 failed
because the arm was not a metered decoder. **This pass is one** --- G0 says so on two independent
measurements --- and judge B still reads the difference *negative*, with its interval clear of
zero. So on AlpacaEval, with a budget calibrated to bind exactly as hard as it does on our own
corpus, **the metered decoder beats selection anchoring.**

| pass | metered arm | `g_sel` | `g_met` | paired `D3` |
|---|---|---|---|---|
| committed, our prompt set | `k=10`, binds `8.376%` | `+0.1065` | `+0.0390` | **`+0.0675 [+0.0330, +0.1020]`** |
| feat-166, AlpacaEval | `k=10`, binds `0.016%` --- INVALID | `+0.0835` | `+0.1792` | `-0.0957 [-0.1124, -0.0792]` |
| **this pass, AlpacaEval** | **`k=1.0`, binds `8.008%`** | `+0.0835` | `+0.1174` | **`-0.0339 [-0.0534, -0.0137]`** |

`g_sel` reads `+0.0835` in both AlpacaEval passes to four decimals, because the selection arm is the
*same generations* judged by the same greedy judge under the same order-averaging --- nothing about
selection changed between them, only the opponent's budget. Constraining the metered arm cost it
`0.0618` of its gain and **it still wins by `0.0339`.**

### What we think this is, stated as a mechanism and not as an excuse

The paper already has the concept: the **support ceiling**. Selection can only serve what the
anchor draws, so its gain is bounded by the anchor's own support, while a metered decoder is
allowed to leave that support at a priced rate. AlpacaEval is instruction-following and the anchor
is `TinyComma-1.8B`, a `1.8`B base model trained on openly licensed text. On this workload the
anchor is out of its depth, so `64` draws from it are `64` draws of the same inadequacy, while
`8%` of steps drifting toward an instruct model is decisive.

That is a measured instance of a limitation the paper states rather than a contradiction of a
claim it makes --- **but the paper does not currently scope its headline to workloads inside the
anchor's support, and after this it must.** The reversal is a claim about a regime, and this arm
measures where the regime ends. We are not able to say from one benchmark where the boundary sits,
and we do not claim to.

We also state what this does **not** touch. Proposition 1 is unconditional: `q(y) <= n p_s(y)` for
any score and any tie rule, so `log n` is still the certificate and is still `54x` smaller than the
metered arm's realised `171.3` nats. What moves is the *utility* side of the comparison, on one
workload, and it moves against us.

### What runs next

Nothing further on this arm tonight. Mixtral is not run, because running it would produce a number
this registration forbids reading and which nobody could unsee. The honest next step is a
**registered** arm asking the scoped question directly --- does the reversal hold on workloads
inside the anchor's support and fail outside it --- with the support measured in advance rather
than inferred after a failure. That arm is not written yet and this file does not pre-empt its
bands.

### CORRECTION, 2026-09-22 09:30 --- the calibration target was wrong by a factor of `1046`

Caught by feat-170's Arm A, whose G0 read `0.00011` activity on the very corpus the target was
supposed to describe. That is the gate doing its job, and the defect is ours.

**`output/phase2/conc_all` is a `k`-SWEEP, not an arm.** It holds six budgets (`0.5` to `20`) and
six prompt classes, including the protected corpus. The target `0.08376` came from scanning the
whole directory --- pooling `k=0.5`, which binds on `48%` of steps, with the protected classes the
judge never reads. **The arm the paper actually judges is `k=10` over the three ordinary classes,
and it reads `24` of `299{,}843` = `0.000080`.**

This is caution (v) verbatim --- *a reference number carries its protocol, and a gate built on one
without it is a gate built on nothing* --- and it was committed **in the same session in which
feat-166's scoring log generalised that caution from reference numbers to reference findings.**
Knowing the rule is not applying it. The repair is structural rather than a corrected constant:
`analysis/budget_calibration.py` no longer accepts a typed target at all; it derives one from the
arm the judge loads, by the same `(k, classes)` selection `analysis/selection_decoding.py` uses.

**What this changes, stated in full.**

1. **G0's activity leg was measured against a number that did not mean what it said.** Its *other*
   leg --- `4.3%` of completions byte-identical to the opponent, against feat-166's `98.6%` ---
   needs no reference and stands: at `k=1.0` the arm genuinely serves text the opponent does not.
2. **With the correct target the `argmin` rule picks `k=10`,** which is feat-166's arm, and
   **G-cal FAILS**, because all five grid points sit above `0.000080`; a grid that does not bracket
   the target reports NOT RUN rather than taking the nearest endpoint. So the calibrated arm this
   registration built is **not** matched to the committed protocol --- it binds about a thousand
   times harder.
3. **feat-166 is therefore re-labelled VALID.** Its INVALID note argued *"the arm labelled metered
   decoder is not one"*; at `k=10` it binds on `0.016%` of steps where the committed arm binds on
   `0.008%`, so it is as metered as the paper's own headline arm and rather more so. **What was
   wrong was the comparison, not the arm.** Its `G0` still FAILED --- judge B read `-0.0957`, where
   the gate required positive --- so B1 remains unread on that arm too, and no band moves.
4. **Nothing about `-0.0339` changes.** It is a within-pass paired difference at a budget that
   binds on `8.008%` of steps; what changes is the sentence describing that budget, which said
   "calibrated to bind as hard as on our own corpus" and should have said "about a thousand times
   harder".

**And the finding hiding under the defect is worth more than the arm was.** At `k=10` the paper's
own metered decoder is active on `0.008%` of its own workload's steps --- one step in twelve
thousand. The realised `171.3` nats against `K = kT_{max} = 2000` already implied it and the paper
reports both, but the activity rate states it in the currency a reader checks: **the paper's
headline opponent is, at its headline budget, the unconstrained risky model to within sampling
noise.** That is the vacuous horn of the paper's own dichotomy, in the paper's own experiment. The
manuscript now says so in `app:workload`.

**The manuscript carried the false number for about five hours** (`4276727` through `09:30`), in
the sentence *"$0.016\%$, against $8.376\%$ for the same $k$ on our workload"*. It is corrected to
the measured `0.008%`, and the paragraph's conclusion is strengthened rather than weakened: the
reversal fails on AlpacaEval at the paper's own budget **and** at one that binds a thousand times
harder, so it cannot be explained away as an artefact of a vacuous meter.
