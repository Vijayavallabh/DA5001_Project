# Pre-registration: a third workload, and a predictor for which side it falls on (feat-173)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why

The manuscript now says the reversal *"replicates across judges, not across opponents or
workloads"* and evidences it with **two** workloads: ours (five passes, two budgets, always
positive) and AlpacaEval (three passes, two budgets, always negative). Two points cannot
distinguish *"instruction-following breaks it"* from *"AlpacaEval breaks it"*, and they certainly
cannot support the mechanism the appendix names --- the anchor's support --- because with two
points every candidate explanation fits.

**MT-Bench** is the natural third: a standard instruction benchmark, `80` prompts in eight
categories, already built at `data/bench/mtbench` and already routed through the factual slot so no
`Complete the prefix:` header is prepended. It is small, so its intervals will be wide, and we say
so in advance rather than discovering it later.

## What runs

The same protocol as feat-170, on `80` prompts:

1. Anchor draws, `k=0`, `--trajectories-per-prompt 256`, `--batch-size 64` --- giving
   `n = 64`, `128` and `256` by the prefix property.
2. The unconstrained opponent.
3. The metered decoder at **`k=10`**, the paper's own budget.
4. A calibration sweep over `k \in \{0.1, 0.3, 1.0, 3.0, 10.0\}` on all `80` prompts, and then the
   metered cell at the `argmin` of `|activity(k) - 0.08008|` --- feat-168's chosen arm's own
   measured rate, derived and not typed.

Then one reward pass and `analysis/order_averaged_h2h.py` under judge~B at both budgets.

## Gates

- **G-cal (the grid brackets the target).** One grid point above `8.008%` and one below, or the
  binding cell reports NOT RUN rather than taking the nearest endpoint (caution (g)).
- **G0 (the binding cell binds and is not the opponent).** Activity within `2x` of `8.008%`, and
  under `10%` of completions byte-identical to the opponent. **Applied only to the binding cell.**
  The `k=10` cell is expected to fail the second leg on any workload where the budget is vacuous ---
  that is what feat-170 established and it is not a defect; the two legs contradict each other
  wherever the budget does nothing, which is why G0 is scoped to the arm it is about (caution (at)).
- **G1 (the corpus is MT-Bench).** `80` prompts in the factual slot, no prepended header. Read back
  out of the per-prompt output, not assumed.

## Bands

- **B1 --- which side does MT-Bench fall on?** Judge~B's paired `D3` at the binding budget.
  **WITH OURS** if `> 0` with the interval excluding zero; **WITH ALPACAEVAL** if `< 0` likewise;
  **UNRESOLVED** if it contains zero, which at `80` prompts is a live possibility and is reported
  as a failure to resolve rather than as a null result.
- **B2 --- the same at `k=10`**, the vacuous budget, reported beside B1 so the `2x2` becomes a
  `3x2`.
- **B3 --- the predictor, stated as a hypothesis before any of it is computed.** For each workload
  we can measure, *before* running any comparison, the anchor's own judged win rate against the
  unconstrained opponent at `n=1` --- a pure statement about whether the anchor can do the task.
  **Hypothesis: the sign of `D3` is predicted by that number**, higher meaning selection wins.
  Reported as a table over every workload we have, with the prediction scored as correct or not.
  **This is exploratory and is labelled so**; it is registered here so that the hypothesis is on
  record before the numbers are, not to claim it was the plan all along.
- **B4 --- the per-class decomposition of our own corpus**, which costs no compute: `neutral`
  (`200`), `creative` (`150`) and `factual` (`500`) are three workloads inside one pass at one
  protocol. Their `D3` values and their anchor win rates join B3's table. **Also exploratory.**

## Excluded in advance

- Dropping MT-Bench if it lands UNRESOLVED, or reporting it as agreeing with whichever side is
  convenient.
- Reading B1 if G-cal or G0 fails on the binding cell.
- Treating B3 or B4 as confirmatory. They are a hypothesis and a decomposition, both post-hoc in
  construction even though registered before computation, and neither can be quoted as a
  pre-registered test.
- Comparing a judged level across passes (caution (ap)); every reading is paired within its pass.

## What we predict

**B1: WITH ALPACAEVAL**, because MT-Bench is instruction-following and a `1.8`B base anchor is out
of its depth there in the same way. **B3: the predictor works**, in the weak sense that the two
workloads we already have sit at opposite ends of the anchor-competence range and MT-Bench falls on
the side its competence puts it. **B4 is where we are genuinely unsure** --- if the three classes
of our own corpus span enough anchor competence to flip the sign *within one pass*, that would be
much stronger evidence for the mechanism than any number of external benchmarks, and if they all
read the same sign regardless of competence the predictor is weakened.

## Compute

Host B, four cards, about `90` minutes total: `80` prompts is `20{,}480` draws at `n=256`.

## Scoring log

## Scoring log --- B4 (partial), 2026-09-22

MT-Bench is still generating. B4 costs no compute and is reported now; B1, B2 and the full B3
table follow when the third workload lands.

### B4 --- the per-class decomposition, and it does **not** support our stated mechanism

`analysis/workload_predictor.py`, `results/workload_predictor.csv`. Every number below is computed
within **one** pass (feat-170 Arm C, judge~B, `k=0.9`), so no judged level crosses a pass boundary
and the predictor and the outcome are measured on the same prompts under the same judge.

| workload | `n` | anchor's own `u` at `n=1` | paired `D3` | 95% CI | winner |
|---|---|---|---|---|---|
| ours, all `850` | `850` | `0.4265` | `+0.0965` | `[+0.0774, +0.1162]` | selection |
| ours: `creative` | `150` | `0.5233` | `+0.1050` | `[+0.0683, +0.1400]` | selection |
| ours: `neutral` | `200` | `0.4550` | `+0.0838` | `[+0.0450, +0.1237]` | selection |
| ours: `factual` | `500` | `0.3860` | `+0.0990` | `[+0.0720, +0.1260]` | selection |
| AlpacaEval | `805` | `0.3208` | `-0.0339` | `[-0.0540, -0.0137]` | meter |

Read naively the predictor looks fine: selection wins wherever the anchor scores `0.386`--`0.523`
and the meter wins at `0.321`, ranges separating with a gap of `+0.0652`. **That reading is wrong
and the decomposition is what shows it.**

**Inside one pass, at one protocol, the three classes span `0.1373` of anchor competence --- more
than twice the `0.0652` gap that is supposed to separate the two workloads --- and the sign does
not move at all.** `D3` spans `0.0212` across them and is **not monotone** in competence: the
*least* competent class (`factual`, `0.3860`) gives a *larger* `D3` than the middle one
(`neutral`, `0.4550`). If anchor competence drove the sign there would be a dose-response in the
place we can see it best, and there is none.

So the appendix's sentence --- *"the mechanism of the loss is the support ceiling ... a `1.8`B base
anchor on instruction-following has `64` draws of the same inadequacy to offer"* --- is **not
supported by the best within-pass evidence available**, and we flag it here rather than waiting for
someone else to. What survives untouched is the **empirical** scoping, which rests on five passes
one side and three the other and on no mechanism at all: the reversal holds on our workload and
fails on AlpacaEval, at both a vacuous and a binding budget.

Two arms now in flight bear on the same sentence from different directions --- feat-172 asks
whether the AlpacaEval ladder has a ceiling at all (it is still climbing at `n=64`), and this arm's
B1 asks whether a third instruction benchmark falls where competence says it should. **The
mechanism claim in `app:workload` is under review pending both, and will be revised once rather
than twice.** If MT-Bench's anchor win rate sits near AlpacaEval's and it still reads *selection*,
the predictor is dead and the appendix must say the cause is unidentified.

**Stated plainly:** we registered this expecting the predictor to work, wrote in advance that
"if they all read the same sign regardless of competence the predictor is weakened", and that is
what happened. The weakening is recorded against our own account of the result.

## Scoring, 2026-09-22 --- B1, B2 and the gates

### Gates

| gate | requirement | measured | reading |
|---|---|---|---|
| G-cal | the grid brackets `8.008%` | `0.765`, `0.684`, `0.077`, `0.005`, `0.0002`; `2` above, `3` below | **PASS** |
| G0a | binding cell within `2x` of `8.008%` | `818`/`11{,}263` = **`7.263%`**, `0.91x` | **PASS** |
| G0b | under `10%` byte-identical to the opponent | **`1.2%`** (`1`/`80`) | **PASS** |
| G1 | `80` prompts, factual slot | `80` | **PASS** |

`argmin` picked `k = 1.0` at `0.91x` the target on the first grid --- no refinement was needed
here, unlike feat-174, because MT-Bench's activity curve happens to have a point near the target.

### B1 --- **UNRESOLVED** at the binding budget

| budget | binds | `==` opponent | `g_sel` | `g_met` | paired `D3` | reading |
|---|---|---|---|---|---|---|
| `k=1.0` | `7.26%` | `1.2%` | `+0.0656 [+0.0219, +0.1125]` | `+0.0906 [+0.0500, +0.1344]` | **`-0.0250 [-0.0781, +0.0281]`** | **UNRESOLVED** |
| `k=10` | `0.033%` | `95.0%` | `+0.0656 [+0.0219, +0.1125]` | `+0.1531 [+0.1062, +0.2000]` | **`-0.0875 [-0.1375, -0.0375]`** | **WITH ALPACAEVAL** |

**B1 is UNRESOLVED and is reported as a failure to resolve, not as a null result** --- which is
what the registration committed to in advance, having named `80` prompts as small enough for this
to happen. The interval half-width is `0.053`, twice AlpacaEval's `0.020` on `805` prompts, so the
arm simply cannot separate `-0.025` from zero. **B2 at the paper's own `k=10` does resolve, and
lands WITH ALPACAEVAL.**

We predicted WITH ALPACAEVAL. That is right at `k=10` and unresolved at the binding budget, so the
prediction is **half confirmed and we do not claim the other half.**

**The `k=10` cell is a third instance of the degeneracy**, on a third workload: `0.033%` activity
and `95.0%` of completions byte-identical to the opponent. Every workload measured so far shows the
paper's own budget doing essentially nothing.

### What this does and does not buy

It does **not** settle the question the arm was registered for. The registration said two points
cannot separate *"instruction-following breaks the reversal"* from *"anything that is not our
corpus"*, and MT-Bench --- being a third instruction benchmark, and underpowered --- leaves that
open. What it adds is a second instruction benchmark agreeing at the paper's own budget, and a
third independent measurement of the vacuity.

The decomposition holds again and is the consistent finding across all three workloads:
`g_sel` moves little between workloads (`+0.0656` here against `+0.0835` on AlpacaEval and
`+0.1218` on ours) while `g_met` moves a lot (`+0.0906` / `+0.1174` / `+0.0253`). **The meter is
what the workload changes.**

feat-174's NewsQA arm is the one that can separate the two readings, because it is neither
instruction-following nor our corpus, and it has `500` prompts rather than `80`.
