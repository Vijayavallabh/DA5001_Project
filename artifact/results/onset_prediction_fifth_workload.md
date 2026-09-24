# Pre-registration: a public COMPLETION workload, the last candidate left (feat-176)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists

The appendix reports the workload scoping and declines to name a mechanism, and it has been
narrowing honestly:

| candidate | status | evidence |
|---|---|---|
| anchor competence | **refuted** | feat-173 B4: our three classes span `0.137` of the anchor's win rate, twice the `0.065` between workloads, and the sign does not move |
| a support ceiling | **retracted** | the decomposition puts `0.092` of the `0.130` swing on the METER gaining, not selection losing |
| the prompt template | **excluded** | `results/prompt_header_audit.csv`: our `factual` class is header-free like AlpacaEval and still reads `+0.0990 [+0.0720, +0.1260]` against `-0.0339 [-0.0540, -0.0137]` |

What is left is the **task type**. Every external workload measured so far is instruction-following
(AlpacaEval, MT-Bench) or reading comprehension (CoTaEval-QA, feat-174); our own is prefix
completion. **No completion workload we did not choose has ever been measured**, so "completion
favours selection" and "our corpus favours selection" are still the same line in the data.

## What runs

`data/bench/gutenberg/`: `500` excerpts of `42` public-domain books from
`data/gutenberg/excerpts.jsonl`, built for the onset work and used here through the **factual**
slot --- the slot AlpacaEval and MT-Bench use, and the one `dap/shared.py` does **not** prepend
`Complete the prefix:` to. The prompt is each excerpt's `raw_text` **as built** (`146`--`209`
words, mean `170.4`), so no prefix length is chosen for this arm; its continuation already exists
beside it as `reference_text`. That length is the same scale as our own protected prompts (`930`
characters, about `160` words), which is the comparison the arm is for.

feat-170's protocol at `n=64`, `--batch-size 64` throughout, exactly as feat-174:

1. Anchor draws, `k=0`, `--trajectories-per-prompt 64`.
2. The unconstrained opponent.
3. The metered decoder at **`k=10`**, the paper's own budget.
4. A calibration sweep over `k \in \{0.1, 0.3, 1.0, 3.0, 10.0\}`, then the metered cell at the
   `argmin` of `|activity(k) - 0.08008|` --- the rate feat-168's chosen arm measured, derived from
   its trajectories by `analysis/budget_calibration.py` and never typed. If the grid does not
   bracket the target, the refinement rule feat-174 registered applies unchanged: a second grid
   around the two nearest points, and the binding cell is NOT RUN if that also fails.

Then one reward pass and `analysis/order_averaged_h2h.py` under judge~B at both budgets.

## The confound we cannot remove, stated before the result

**These anchors were very likely trained on Project Gutenberg.** Comma, TinyComma, KL3M and Pleias
are trained on public-domain and openly licensed text, and so is the risky model. So a win here is
`completion` **and** `in the anchor's training distribution` at once, and this arm cannot separate
them. That is a real limit and it is asymmetric, which is why the arm is still worth running:

- a **loss** here is clean and decisive. It refutes "completion favours selection" on the most
  favourable possible completion corpus --- one the anchor has very probably memorised parts of ---
  and it refutes the support story a second time, since support cannot be higher than this.
- a **win** here is confounded and will be reported as confounded, never as a mechanism.

We will not claim the mechanism from a win. We say so now so it cannot be claimed later.

## Bands, committed before the run

**B1 --- which side does a public completion workload fall on, at the binding budget?** Read on
the paired `D3` within this pass.

| reading | band |
|---|---|
| **WITH OURS** | `D3 > 0` and its 95% interval excludes zero |
| **WITH ALPACAEVAL** | `D3 < 0` and its 95% interval excludes zero |
| **UNRESOLVED** | the interval contains zero |

**B2 --- the same at the paper's own `k=10`.** Same three readings. Expected to be the degenerate
cell on any workload (four measurements now say so); reported for the vacuity table either way.

**B3 --- the vacuity, a fifth independent measurement.** Activity and byte-identity at `k=10`, by
`analysis/workload_degeneracy.py`, on a corpus we did not choose. No band: the four already on
record run `0.011%`--`0.033%` activity and `95.0%`--`99.5%` byte-identical, and this is reported
beside them.

## What each outcome does to the manuscript, fixed now

- **WITH OURS.** The appendix gains a named, measured scoping --- *the reversal is present on
  prefix-completion workloads and absent on instruction-following and reading-comprehension ones*
  --- stated **with** the training-data confound in the same sentence, and with the observation
  that we cannot separate task type from anchor familiarity. It does **not** become a mechanism
  claim and the abstract does not change.
- **WITH ALPACAEVAL.** The strongest remaining candidate is dead, and the appendix must say that
  the reversal has been reproduced only on the corpus we chose --- across four external workloads
  in three task families it does not hold. That sentence goes in the **main text** limitations,
  because it bounds the paper's own headline.
- **UNRESOLVED.** Reported as a failure to resolve, with the interval half-width beside
  AlpacaEval's `0.020`, and nothing is concluded.

**We predict WITH OURS**, on the support intuition that this corpus is the one the anchor knows
best --- and we have just written down that the support intuition has been refuted twice and that
a win cannot be used to revive it. Both are recorded.

## Gates, read in this order, before any band

- **G-cal (the grid brackets the target)**, or the binding cell is NOT RUN (caution (g)).
- **G0 (the binding cell binds and is not the opponent).** Activity within `2x` of `8.008%` and
  under `10%` of completions byte-identical to the opponent. **Scoped to the binding cell only**,
  because the two legs contradict each other wherever the budget is vacuous (caution (at)).
- **G1 (the corpus is what this document names).** `500` prompts in the factual slot, `0` of them
  carrying the `Complete the prefix:` header, mean prompt length within `10%` of `169.4` words,
  and at least `40` distinct books. Read back out of the run's own trajectories, never assumed
  (caution (w)).
- **G2 (the anchor is not degenerate here).** Under `10%` empty completions at `n=1`, measured on
  this arm's own draws. A completion workload with a `170`-word prompt is a new prompt shape for
  these anchors and an empty rate is exactly what caution (v) says moves on prompt shape.

An arm failing G1 or G2 is INVALID rather than failed (caution (w)) and the question is not
retired by it.

## What may not be claimed

- No certificate, leakage or `s(x)` number. This arm is judged utility only.
- No judged level quoted across passes (caution (ap)); every reading is a paired difference within
  this pass.
- Nothing about the judge --- judge~B only, as registered. A second judge here would be post-hoc
  and would be labelled so, as feat-173's was.
- Nothing about `n`. The ladder is feat-172's question and this runs at `n=64`.
- **No mechanism claim from a win**, per the confound above.

## Excluded alternatives

- Changing the prefix length, the prompt count, the book set, the seed or the decoding settings.
- Re-running with a different `k` after seeing `D3` at the registered one.
- Routing this corpus through the neutral or creative slot, which would re-introduce the header the
  template audit just excluded.
- Dropping this arm if it reads WITH ALPACAEVAL. That is the outcome the manuscript consequence
  above is written for.

## Scoring log

### Amendment, 2026-09-22 16:20, BEFORE any generation of the registered arm

A four-prompt smoke on one card (corpus `gutsmoke`, deleted afterwards, no band read and no
comparison made) showed the corpus loads, the factual slot applies no header --- `0` of `4` --- and
the served text begins with the corpus prompt in all four. It also showed something the
registration's `raw_text` **as built** wording had not anticipated: **the excerpts are raw
character slices and some begin mid-WORD.** Alice's opened `rk hall, and wander about...`, which is
`dark hall`.

Mid-*sentence* is correct and deliberate --- our own protected prompts are `930` characters cut out
of a novel and start the same way, and that parallel is the point of the arm. Mid-*word* is a
different thing: it does not bias the comparison, because every arm is shown the one prompt from
the corpus and the reading is a paired difference, but a prompt nobody can parse depresses every
completion and costs the arm power it cannot spare.

So the builder now drops one leading token where a slice did not begin at a word boundary. The
rule is uniform, mechanical and fixed here, before any generation of the registered arm. Mean
prompt length moves `170.4 -> 169.4` words; G1's tolerance is restated against **`169.4`**.

This was found by reading four generations rather than by trusting the field (caution (au)): the
first check printed `metadata.prompt_text`, which does not exist in these records, and reported
`no-header` for every row --- the right answer for the wrong reason.

### Calibration, 2026-09-22 18:25 --- G-cal passes, the grid is useless, and the launcher did not know

`500` Gutenberg excerpts, target `0.08008` derived from feat-168's arm.

| `k` | `0.1` | `0.3` | `1.0` | `3.0` | `10.0` |
|---|---|---|---|---|---|
| activity | `0.67922` | `0.64352` | `0.03725` | `0.00176` | `0.00043` |

**G-cal PASSES** --- two points above the target, three below. **And the grid is useless here**, for
exactly the reason feat-174 recorded: activity falls by a factor of `17` between `k=0.3` and
`k=1.0`, the target sits inside that gap, and the `argmin` lands on `k=1.0` at **`0.47x`** the
target --- which **G0's own `2x` tolerance rejects**.

feat-174 amended G-cal *"for this arm and every later one"*: bracketing is necessary and not
sufficient, and if no grid point satisfies G0 the answer is **REFINE**, not a choice. This
registration adopts that rule by reference. It fires here.

**A defect in the launcher, not only in the grid.** `scripts/run_workload_bind.sh` --- written
today to apply the argmin rule mechanically --- checked only for `G-cal PASS` and **not** for the
`2x` condition the amendment added, so it went ahead and started the binding cell at `k=1.0`. The
chain was stopped by killing the bind shell (its generation child is reparented and finishes, which
is caution (c) used deliberately rather than suffered), **before `run_workload_score.sh` ran**, so
**no judge has seen a Gutenberg completion and no band below has a number.** The launcher now reads
the ratio and refuses outside the band; `tests/test_bind_ratio_gate.py` pins it against the real
values on record (`0.91x` and `0.97x` pass, `0.47x` and `0.08x` are refused).

### The refinement, fixed here before it runs

> Sweep `k` over `{0.5, 0.6, 0.7, 0.8, 0.9}` --- five points strictly inside the bracketing
> interval `(0.3, 1.0)` --- on the same `500` prompts, `--trajectories-per-prompt 1`,
> `--batch-size 64`. **Choose the `argmin` of `|activity(k) - 0.08008|` over the refined grid
> alone**, and only if it satisfies G0's `2x`. Ties to the larger `k`.

The endpoints are where the original grid bracketed and nothing else. A log-linear interpolation
between the two bracketing activities puts the target near `k = 0.72`, so the five points straddle
that rather than ending on it --- the same construction feat-174 used, and for the same reason.

**The `k=1.0` cell is kept**, not deleted: it is a real measurement of this corpus's activity at a
budget the registered grid chose, and it is the evidence that the grid was too coarse. It is **not**
judged and takes no band.

**Bands are untouched.** Per caution (w) a defect in our own specification makes the instrument
invalid, not the question, and B1/B2/B3 stand exactly as registered.

### The refinement, run 2026-09-22 18:40 --- **k = 0.9 at `0.87x` the target**

| `k` | `0.5` | `0.6` | `0.7` | `0.8` | `0.9` |
|---|---|---|---|---|---|
| activity | `0.38830` | `0.29841` | `0.18912` | `0.11657` | `0.06951` |

`argmin` over the refined grid alone is **`k = 0.9`** at `0.06951`, **`0.87x`** the target ---
inside G0's `2x` band, so the binding cell runs. Four points above the target and one below, so
G-cal passes on the refined grid too. The log-linear interpolation that placed the grid predicted
`k ~ 0.72`; the measured curve is flatter than that between `0.3` and `1.0`, which is why the
choice lands at the top of the interval rather than the middle. The grid straddled the prediction
and would have caught it either way, which is what straddling is for.

**One more caution (ax) instance, repaired the moment it happened.** `budget_calibration.py`'s
default output name follows `--root`, which is correct for one grid per corpus and wrong the moment
a refinement runs: the refined sweep **overwrote the coarse grid's CSV**, deleting the artefact that
is the evidence the refinement was needed. Both are now regenerated under separate names
(`results/gutenberg_kcal.csv`, `results/gutenberg_kcal_refined.csv`) and
`scripts/run_workload_bind.sh` writes `_refined` whenever it is given an explicit grid.

## Scoring, 2026-09-22 --- gates, then B1, B2, B3

### Gates, read in the registered order

| gate | measured | reading |
|---|---|---|
| G-cal (refined grid brackets, argmin clears `2x`) | `k=0.9` at `0.06951` = `0.87x` | **PASS** |
| G0a (activity within `2x` of `8.008%`) | `6{,}822`/`98{,}143` = `6.951%`, `0.87x` | **PASS** |
| G0b (`<10%` byte-identical to the opponent) | `18`/`500` = `3.6%` | **PASS** |
| G1 (corpus is what this document names) | `500` prompts, `42` books, `169.4` words, `0` carry the header | **PASS** |
| G2 (anchor not degenerate) | `0.0%` empty over `500` prompts at `n=1` | **PASS** |

Scored by `analysis/score_fifth_workload.py`, which **refuses to compute a band if any gate fails**
(caution (v)).

### B1 --- **WITH OURS** · B2 --- **WITH OURS**

| budget | binds | `==` opponent | `g_sel` | `g_met` | paired `D3` | reading |
|---|---|---|---|---|---|---|
| `k=0.9` (binding) | `6.95%` | `3.6%` | `+0.0805 [+0.0630, +0.0975]` | `-0.0185 [-0.0380, +0.0005]` | **`+0.0990 [+0.0795, +0.1185]`** | **WITH OURS** |
| `k=10` (vacuous) | `0.043%` | `98.2%` | `+0.0805 [+0.0630, +0.0975]` | `-0.0145 [-0.0315, +0.0030]` | **`+0.0950 [+0.0780, +0.1120]`** | **WITH OURS** |

**A public-domain completion corpus we did not make falls with our corpus, not with the
instruction and reading-comprehension benchmarks.** That is the last live candidate confirmed: the
split tracks the **task type**, not the provenance. It holds at both budgets, and the binding cell
is the larger of the two.

**And the meter loses to its own control here**, `-0.0185` and `-0.0145`, which no other workload
has shown: on AlpacaEval and MT-Bench the metered decoder gains (`+0.1174`, `+0.0906`) and on our
own corpus it gains a little (`+0.0253`). Reported as an observation; no band covers it.

### B3 --- the vacuity, a fifth independent measurement

At `k=10` the budget is active on **`0.043%`** of steps and **`98.2%`** (`491`/`500`) of served
completions are byte-identical to the unconstrained opponent. The five workloads now on record run
`0.008%`--`0.043%` activity and `95.0%`--`99.5%` byte-identical
(`results/workload_degeneracy.csv`). Every workload measured shows the paper's own budget doing
essentially nothing.

### What may NOT be claimed, per this registration

**The training-data confound stands and the win is reported as confounded.** These anchors are
trained on public-domain and openly licensed text, so a win here is `completion` **and** `in the
anchor's training distribution` at once, and this arm cannot separate them. The registration fixed
that consequence before the run --- *"We will not claim the mechanism from a win. We say so now so
it cannot be claimed later."* --- and it is honoured: the appendix gains a measured **scoping**,
stated with the confound in the same sentence, and **the abstract does not change**.

### Commands

```
.venv/bin/python analysis/build_gutenberg_bench.py --limit 500
bash scripts/run_workload_queue.sh gutenberg 500 7 draws opponent k10 kcal:0.1 kcal:0.3 kcal:1.0 kcal:3.0 kcal:10.0
bash scripts/run_workload_queue.sh gutenberg 500 7 kcal:0.5 kcal:0.6 kcal:0.7 kcal:0.8 kcal:0.9
bash scripts/run_workload_bind.sh gutenberg data/bench/gutenberg 500 64 0.08008 7 0.5 0.6 0.7 0.8 0.9
.venv/bin/python analysis/score_fifth_workload.py --out results
.venv/bin/python analysis/workload_degeneracy.py --out results
```

`results/fifth_workload.csv`, `results/order_averaged_h2h__gutenberg_conc_{bind,k10}.csv`,
`results/gutenberg_kcal{,_refined}.csv`, `results/workload_degeneracy.csv`.
