# Pre-registration: where does best-of-n stop turning over? The scorer-scale boundary

Committed **before the arm runs**, as every arm in this line has been. Nothing above the
`## Scoring log` rule may be edited after the run; the scoring log is appended underneath.

## Why this arm exists

feat-116 measured something nobody registered and nobody expected. Sweeping `n` with a
Qwen2.5-7B reward raises judged gain monotonically, `+0.0195` at `n=2` to `+0.1075` at `n=64`.
Sweeping `n` with a Qwen2.5-0.5B reward **peaks at `n=16` and falls**, ending at `+0.0220` with four
times the draws it had at its peak. The reading we offered is that taking the argmax of a weak score
over a larger pool selects increasingly on the score's noise, so `log n` is not a free knob: the
certificate keeps improving in `n` while the utility bought with it turns over.

That claim currently rests on **one scorer at one scale**, and it was not tested against an
interval even at that scale --- feat-116 committed no band on monotonicity, so the fall from
`+0.0395` to `+0.0220` is a curve we read off a table. It is also the paper's most
deployer-relevant sentence, because a deployer picking a reward model is choosing exactly the
quantity this turnover depends on, and the paper currently gives them a direction and no number.

This arm asks the only question that converts it into one: **between `0.5`B and `7.6`B, where does
the turnover stop?** Two intermediate scorers, `Qwen2.5-1.5B-Instruct` (`1.5437`B measured) and
`Qwen2.5-3B-Instruct` (`3.0859`B measured), from the same family with the same chat template and the
same `log p("Yes") - log p("No")` reward on the same fixed template, so across all four arms scale
remains the only thing that varies.

## A correction this arm surfaced before it ran, applied first

The cost model priced its models by their **names**. Counted off the loaded checkpoints,
Qwen2.5-7B-Instruct holds `7.6156`B parameters and not `7.0`, TinyComma `1.7586`B and not `1.8`, and
Llama-3.1-8B-Instruct `8.0303`B. Selection's serving cost was therefore understated by `8.8%`
wherever it appeared: the published `57.5\times` at `n=64` is **`61.3\times`** and the `7.2\times` at
`n=8` is **`7.7\times`**. `analysis/serving_cost.py` now carries measured counts, the manuscript
carries the corrected ratios, and the correction runs against us. It is recorded here rather than
quietly, because feat-116's own docstring warned about exactly this ("0.494B measured, not the
label") and we made the mistake anyway on the three constants we did not re-measure.

## Design

**No generation.** Re-score the same 32,000 cached candidates in `output/phase5/sel_anchor64` with
the two new scorers, then judge **all four scorers in one pass** together with the metered decoder
and both controls. Judging all four together is not optional: this paper's own instrument checks
forbid quoting judged levels across passes, and the comparison here is *between scorers*, so they
must share a pass, a control, an opponent and a prompt set.

Protocol is feat-113's corrected one, unchanged: one true prompt per item from
`dap.shared.load_prompt_corpus`, every item judged in **both** presentation orders and averaged, one
fixed opponent, judge B (`microsoft/Phi-3.5-mini-instruct`), 10,000-resample bootstrap paired over
the shared prompts. Arms nest by seed order, so each distinct served completion is judged once per
order and the arms are assembled from those utilities.

Grid `n \in {2,4,8,16,32,64}` for each of the four scorers, plus `sel_n1` (the rank-0 draw, the
shared control), `metered_k10` and `anchor_k0`.

## The estimand

For a scorer `s`, write `g_s(n)` for its order-averaged judged gain over `sel_n1`. The turnover
statistic is the **terminal drop**

    D_s = g_s(64) - g_s(16) ,  paired per prompt.

`n=16` is fixed by feat-116, which is a *prior* experiment: for the two new scorers this is an
out-of-sample comparison and not a cell chosen after looking. For the 0.5B scorer it is the same
cell feat-116 read off a table, now given an interval for the first time.

## Bands, committed before the run

* **G0 -- the replication gate, scored first.** `sel7b_n64`, `sel05b_n64` and `metered_k10` must
  land within `+/-0.04` of feat-116's `+0.1075`, `+0.0220` and `+0.0400`. If any misses, **the arm
  is reported as a failed replication and no band below is quoted.**
* **G1 -- the terminal drop, one reading per scorer.** `TURNS OVER` if `D_s < 0` and its 95% CI
  excludes 0; `FLAT` if the CI contains 0; `RISES` if `D_s > 0` and its CI excludes 0.
  Committed expectations: `0.5`B **TURNS OVER**, `1.5`B **FLAT**, `3`B **RISES**, `7`B **RISES**.
* **G2 -- the shape statistic.** Spearman between gain and `log n` over the six grid points, per
  scorer. `MONOTONE` at `\rho = 1.0` exactly, else the value is reported as a description. Six
  points cannot carry a useful interval and none is claimed. Committed expectation: monotone at
  `3`B and `7`B, not at `0.5`B.
* **G3 -- the boundary, which is the number the paper wants.** The smallest scorer whose G1 reading
  is not `TURNS OVER`.
  * `BOUNDARY AT 1.5B` / `BOUNDARY AT 3B` / `BOUNDARY AT 7B`
  * `NO TURNOVER FOUND` -- not even the `0.5`B scorer reads `TURNS OVER` under its interval.
  Committed expectation: **`BOUNDARY AT 1.5B`**.
* **G4 -- does the terminal gain rise with scorer scale?** `g_s(64)` across the four scorers, with
  the three adjacent paired contrasts. Descriptive; no threshold is committed, because four points
  cannot carry a law and we will not claim one.

## What each reading costs, committed in advance

* **`BOUNDARY AT 1.5B` or `BOUNDARY AT 3B`** -- the appendix reports the boundary and Limitations
  replaces its present direction-only sentence with the measured scale. The paper states it as a
  boundary measured on four scorers of one family on one workload, **not** as a law, and says in
  those words that it does not know whether the threshold is a parameter count, a capability level
  or a property of this reward template.
* **`BOUNDARY AT 7B`** -- stronger and worse for the mechanism: a scorer must be comparable in size
  to the risky model before `n` is safe to raise, which makes the compute problem structural rather
  than incidental. Limitations says so, and the `61.3\times` concession is then the *floor* of what
  the mechanism costs, not a point on a curve one can walk down.
* **`NO TURNOVER FOUND`** -- feat-116's observation does not survive an interval. The appendix
  sentence "peaks at `n=16` and falls" is **retracted and replaced** with the interval-tested
  statement, the `log n`-is-not-a-free-knob claim comes out of the paper and out of the handoff, and
  this pre-registration is cited as what retracted it.
* A `FLAT` reading at every scorer including `0.5`B is `NO TURNOVER FOUND` and costs the same.

## Alternatives excluded in advance

* No scorer outside the Qwen2.5 instruct family, and no fifth scale added after seeing the result
  to bracket a boundary more tightly. If the boundary falls between two of these four, the paper
  says it falls between them.
* No other judge; judge C remains excluded, being the opponent's own checkpoint.
* No other `n`, and no re-generation: the candidate pool is `output/phase5/sel_anchor64` as it
  stands.
* `D_s` will not be swapped for a different contrast (peak-relative, endpoint-relative, or a fitted
  trend) if `g(64) - g(16)` gives a null. The peak-relative drop is reported as a secondary
  description only, and is explicitly **not** promotable: its peak is chosen by the data.
* The single-order statistic will not be promoted under any reading.

## Scoring log
