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
  carrying the `Complete the prefix:` header, mean prompt length within `10%` of `170.4` words,
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
