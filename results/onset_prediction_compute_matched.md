# Pre-registration: does a 0.5B scorer buy the certificate at the metered decoder's compute?

Committed **before the arm runs**, as every arm in this line has been. Nothing above the
`## Scoring log` rule may be edited after the run; the scoring log is appended underneath.

## Why this arm exists

The v9 draft conceded, on a reviewer's objection and our own arithmetic, that selection anchoring
is expensive to serve. `results/serving_cost.csv` prices it in the deployer's currency rather than
the certificate's: at `n = 64` with the Qwen2.5-7B reward, selection costs **57.5x** the metered
decoder's forward-pass FLOPs, and at `n = 8` it costs **7.2x**. The paper states this and leaves it
there. It is now the weakest point a reviewer can push on, because the nats axis on which selection
wins by `54x` is the axis we chose and the compute axis is the one a deployer pays.

The same CSV contains the arithmetic that could answer it. The 7B reward model, not the anchor,
dominates the cost: at `n` the mechanism runs `n(P_s + P_f)(L_p + T)` and `P_f = 7.0` against
`P_s = 1.8`. A **0.5B scorer** would change the price by a factor of `3.84` at every `n`. That row
is in the CSV today marked `0.5B, NOT RUN`, and the manuscript labels it arithmetic. **An
unmeasured counterfactual in a paper about measuring what certificates actually buy is exactly the
thing this paper exists to object to.** This arm runs it.

The scorer is `Qwen/Qwen2.5-0.5B-Instruct`, **0.494B parameters measured** (not the label): the same
family, the same chat template and the same `log p("Yes") - log p("No")` reward on the same fixed
template as the 7B scorer, so what varies between the two arms is scale and nothing else. A
protocol smoke test on 16 cached candidates confirmed the Yes/No ids resolve and the reward has
spread (sd `1.73`, 12 distinct values of 16) before any band below was written. No gain was
computed in that check.

## What is on trial, and what is not

The certificate is **not** on trial and cannot be: `q(y) <= n p_s(y)` holds for *any* score, so
`D_inf(q||p_s) = log n` exactly as before and a 0.5B scorer buys the identical `1.386` nats at
`n = 4` that a 70B scorer would. That invariance is the whole point of Proposition 1 and it is why
this experiment is possible at all. **What is on trial is utility: whether a scorer small enough to
make the mechanism cheap can still find the good draw.**

The leakage axis is also not re-run here, deliberately. `analysis/selection_extraction.py` selects
by the *memorising model's own likelihood* -- an adversarial worst-case score, strictly stronger
than any reward model -- and reports `0.0000` near-verbatim recall at every anchor and every
`n <= 64`. Re-running it with a 0.5B reward would be weaker evidence than what is already on
record, not additional evidence.

## Design

**No generation.** Phase 1 re-scores the 32,000 cached candidates in `output/phase5/sel_anchor64`
(500 prompts x 64 anchor draws) with the 0.5B reward and caches them to
`results/selection_rewards64_qwen05b.csv`. Phase 2 judges the served completions.

The judging protocol is **feat-113's corrected one**, not the protocol that produced
`results/selection_scaling.csv`:

* **One true prompt per item, for every arm**, taken from `dap.shared.load_prompt_corpus` and not
  from `served_prompt()`, which reconstructs it as served-text-minus-generation and disagrees
  across arms in 455 of 500 cases (caution (aa)).
* **Both presentation orders**, utilities averaged per item, so position is removed by
  construction rather than in expectation. The judge's order consistency on this workload is
  `0.24`-`0.35`, UNUSABLE, which is why no level is reported and only gains over a shared control
  are.
* **One fixed opponent** for every arm: the unconstrained risky model at its lowest seed, which is
  what `selection_decoding.load_baseline` returns.
* Judge B, `microsoft/Phi-3.5-mini-instruct`, template and 4-token greedy decode unchanged from
  every judged arm in this paper. Judge C is not used here and will not be substituted: it is the
  risky model's own checkpoint (caution (aa)).
* Bootstrap over prompts, 10,000 resamples, paired over the 500 shared prompts.

**Arms.** Both scorers over the full nested grid, plus the metered decoder and both controls:

| arm | scorer | serving cost vs metered |
|---|---|---|
| `anchor_k0` | -- | 0.18x (the anchor alone) |
| `metered_k10` | -- | **1.00x** |
| `sel_n1` | -- (rank-0 draw; the selection control) | 0.23x |
| `sel05b_n{2,4,8,16,32,64}` | Qwen2.5-0.5B | 0.47x, **0.94x**, 1.87x, 3.74x, 7.49x, 14.98x |
| `sel7b_n{2,4,8,16,32,64}` | Qwen2.5-7B | 1.80x, 3.59x, 7.18x, 14.37x, 28.73x, 57.47x |

Arms nest by seed order, so many `(prompt, n)` cells serve the same completion; each distinct
served completion is judged once per order and the arms are assembled from those utilities.

`sel_n1` is the control for every selection arm and `anchor_k0` for the metered decoder, exactly as
in feat-113, so this arm's `sel7b_n64` and `metered_k10` numbers are a **replication** of feat-113
under the same protocol and must reproduce within the judge's own noise.

## Bands, committed before the run

* **F1 -- does a 0.5B scorer carry any gain at all?** `sel05b_n64` gain over `sel_n1`,
  order-averaged. `WORKS` if its 95% CI excludes 0, `FAILS` if it contains 0.
  Committed expectation: **WORKS**, in `[0.03, 0.09]`.
* **F2 -- what does the scorer's scale cost?** `g(sel05b_n64) - g(sel7b_n64)`, paired.
  `CHEAP` if the CI contains 0; `COSTLY` if the CI excludes 0 and the difference is negative;
  `BETTER` if it excludes 0 and is positive.
  Committed expectation: **COSTLY**, in `[-0.07, -0.02]`.
* **F3 -- the crossover cost, which is the number the paper needs.** The smallest `n` on the 0.5B
  grid whose order-averaged gain has a CI excluding 0 *and* a point estimate at or above the
  metered decoder's own order-averaged gain in this same pass, reported as its serving-cost ratio.
  * `BELOW THE METER` -- ratio `<= 1.0`
  * `WITHIN 4x` -- ratio `<= 4.0`
  * `WITHIN 15x` -- ratio `<= 15.0`
  * `NO CROSSING` -- no grid point qualifies.
  Committed expectation: **`WITHIN 4x`**, at `n = 8` or `n = 16`.
* **F4 -- the compute-matched head-to-head.** `g(sel05b_n4) - g(metered_k10)`, paired, at
  `0.94x` the metered decoder's cost. `MATCHED-COMPUTE WIN` if positive with a CI excluding 0,
  `PARITY` if the CI contains 0, `LOSS` if negative with a CI excluding 0.
  Committed expectation: **PARITY**. With the *7B* scorer `results/selection_scaling.csv` puts
  `n = 4` at `+0.032 [-0.005, +0.070]` on this judge -- an interval already containing zero -- and a
  14x smaller scorer is not expected to do better. I do not expect to win this cell and am running
  it because it is the cell a reviewer will ask about.
* **F5 -- replication.** `sel7b_n64` and `metered_k10` must land within `+/-0.04` of feat-113's
  `+0.1045` and `+0.0400`, the cross-pass floor this paper has measured for itself. If either
  misses, **the whole arm is reported as a failed replication and no band above is quoted**, because
  a pass that cannot reproduce its own reference cannot price a new one.

## What each reading costs, committed in advance

* **`BELOW THE METER`** -- Limitations stops conceding the compute axis. The paper states that
  selection anchoring reaches more judged utility than the metered decoder at lower FLOPs *and*
  `1/124` of the divergence, and the frontier figure carries both axes.
* **`WITHIN 4x` or `WITHIN 15x`** -- the abstract does **not** claim compute parity. Limitations
  replaces the bare `57.5x` with the measured crossover: a smaller scorer moves the price from
  `57.5x` to the measured ratio, and the mechanism is cheaper than the paper said but not free.
  This is the outcome I expect and it is still worth the run, because `57.5x` is currently the only
  compute number in the paper and it is the worst point on the curve.
* **`NO CROSSING`, or `F1 = FAILS`** -- the paper says so in Limitations in those words: the gain
  is a property of the scorer's capability and does not survive shrinking it, so the compute cost
  is intrinsic at the scales tested and is the mechanism's main open problem. The `57.5x`
  concession stands exactly as written and the counterfactual row is removed from
  `serving_cost.csv` rather than left as an unrealised promise.
* `F2 = COSTLY` is **not** a failure under any reading; it is the quantity this arm exists to
  measure.

## Alternatives excluded in advance

* No third scorer will be tried to find one that crosses lower. If the 0.5B scorer does not cross,
  that is the answer.
* No other judge, and no judge C.
* The single-order statistic will not be promoted to primary if the order-averaged one
  disappoints; feat-113 established that the single-order number on these arms is contaminated by
  the `served_prompt()` defect.
* No re-generation: the candidate pool is `output/phase5/sel_anchor64` exactly as it stands, and no
  prompt is dropped for any reason other than absence from an arm.
* The cost model is not renegotiated after seeing the utilities. FLOPs at `2PL` per forward pass is
  the committed currency; the wall-clock caveat (decode steps are bandwidth-bound and prefills are
  compute-bound, which makes a FLOP ratio *over*state selection's latency cost) is stated as a
  caveat in the paper and is not measured here and not claimed.

## Scoring log
