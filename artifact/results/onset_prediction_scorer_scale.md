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

## Scoring, 2026-09-15

**Run.** `output/logs/scorer_scale.log`, GPU 0, 2026-09-15, **`0.53` GPU-h measured** (`compute_hours.csv`; I wrote `2.0` from an estimate before reading the log, which is the mistake caution (v) is about --- a number quoted without opening the record that produces it). Phase 1 re-scored the 32,000
cached candidates with the 1.5B and 3B rewards; phase 2 judged **4,361 distinct served completions**
across 24 selection arms plus the metered decoder and both controls, each in both orders --
**10,722 judged calls**, no generation.

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/scorer_scale.py --out results
```

**G0 REPLICATES, and by a margin worth recording.** The gate allows `+/-0.04`, the judge's own
cross-pass floor. The three reference arms landed within **`0.001`**: `sel7b_n64` `+0.1065` against
feat-116's `+0.1075`, `sel05b_n64` `+0.0230` against `+0.0220`, `metered_k10` `+0.0400` against
`+0.0400`. A protocol that reproduces three arms to a thousandth across two independent passes is
the strongest evidence this paper has that its judged gains, taken over a shared control with
position removed by construction, are stable even though its judged *levels* are not.

| band | value | 95% CI | reading |
|---|---|---|---|
| G1 terminal drop, `0.494`B | `-0.0160` | `[-0.0340, +0.0010]` | **FLAT** |
| G1 terminal drop, `1.5437`B | `+0.0255` | `[+0.0070, +0.0440]` | **RISES** |
| G1 terminal drop, `3.0859`B | `+0.0330` | `[+0.0155, +0.0505]` | **RISES** |
| G1 terminal drop, `7.6156`B | `+0.0445` | `[+0.0270, +0.0620]` | **RISES** |
| G2 Spearman(gain, log n) | `0.5429` / `1.0` / `1.0` / `1.0` | -- | NOT MONOTONE / **MONOTONE** x3 |
| G3 the boundary | -- | -- | **NO TURNOVER FOUND** |
| G4 `1.5`B over `0.5`B at n=64 | `+0.0700` | `[+0.0500, +0.0900]` | **SEPARATES** |
| G4 `3`B over `1.5`B at n=64 | `+0.0040` | `[-0.0145, +0.0225]` | -- |
| G4 `7.6`B over `3`B at n=64 | `+0.0095` | `[-0.0090, +0.0285]` | -- |

**My committed expectation was `BOUNDARY AT 1.5B`. The reading is `NO TURNOVER FOUND`, and its
committed consequence is a retraction.**

## The retraction, executed

feat-116 reported, as an unregistered observation, that the 0.5B scorer "peaks at `n=16` and falls"
and drew from it that "`log n` is not a free knob: the certificate keeps improving in `n` while the
utility bought with it turns over". **That does not survive an interval.** The terminal drop we
registered to test it, `g(64) - g(16)`, is `-0.0160 [-0.0340, +0.0010]` --- negative, and its
interval includes zero, by one thousandth. Under the rule committed before this ran, the 0.5B
scorer reads `FLAT`, not `TURNS OVER`.

So: **the sentence is withdrawn from the appendix and from the handoff, and this file is what
withdrew it.** What replaces it is the interval, stated with its sign and its width, and the honest
description that the fall is suggestive and unestablished. We are not entitled to the deployer-facing
warning we drew from it, and we do not make it.

What *is* in the data, and is reported as description rather than as the registered test: the shape
contrast is stark. Spearman between gain and `log n` is `0.5429` for the 0.5B scorer and exactly
`1.0` --- strictly monotone across all six grid points --- for each of `1.5`B, `3`B and `7.6`B. One
curve out of four is not monotone in this sample. That is a fact about the sample; the pre-registered
test of whether its endpoint is genuinely below its interior did not reach significance, and a
Spearman over six points carries no interval worth quoting. Both halves belong in the paper and
neither alone does.

## What the arm found instead, which is larger than what it was built to test

**Scorer capability saturates, and it saturates early.** The three adjacent contrasts at `n=64`:

| step | difference | 95% CI | |
|---|---|---|---|
| `1.5`B over `0.5`B | `+0.0700` | `[+0.0500, +0.0900]` | separates |
| `3`B over `1.5`B | `+0.0040` | `[-0.0145, +0.0225]` | does not |
| `7.6`B over `3`B | `+0.0095` | `[-0.0090, +0.0285]` | does not |

Going from `0.5`B to `1.5`B buys almost the entire gap. Going from `1.5`B to `7.6`B --- a `4.9x`
larger scorer --- buys `+0.0135` and neither step of it separates. The scorer must clear a bar; past
that bar, more scorer is not measurably more utility on this workload.

**The cost consequence is the largest single improvement to this paper's weakest axis, and no band
was committed on it.** Reported as post hoc, with the mechanical crossing rule taken verbatim from
feat-116's F3 and applied unchanged to scorers feat-116 did not have:

| scorer | gain at n=64 | cost | first cell at or above the meter |
|---|---|---|---|
| `0.494`B | `+0.0230` | `14.73x` | none |
| `1.5437`B | `+0.0930` | **`21.59x`** | `n=8`, **`2.70x`** |
| `3.0859`B | `+0.0970` | `31.67x` | `n=8`, `3.96x` |
| `7.6156`B | `+0.1065` | `61.29x` | `n=16`, `15.32x` |

A `1.5`B scorer reaches **`87.3%`** of the `7.6`B scorer's gain at **`35.2%`** of its serving cost.
The concession the paper has been making --- `61.3x` --- is the cost of the scorer we happened to
use, not the cost of the mechanism.

**One instability we will not hide.** The crossing *cell* is not stable across passes for the 7.6B
scorer: feat-116 put it at `n=8` and `7.66x` (`+0.0415`), this pass puts it at `n=16` and `15.32x`
because `sel7b_n8` read `+0.0360` here against `+0.0415` there. That difference, `0.0055`, is well
inside the judge's `+/-0.04` floor, but it straddles the meter's `+0.0400` and so moves the cell the
rule selects. **A crossing cell is a thresholded statistic and inherits none of the stability the
gains themselves showed under G0.** The paper therefore quotes the `n=64` comparison, which moved by
`0.001` between passes, and quotes any crossing with this caveat attached.

Without a judge, `results/scorer_scale_agreement.csv` (post hoc, no band) gives the same ordering in
reward space: against the 7.6B reference the within-prompt Spearman is `0.1333` at `0.5`B, `0.4185`
at `1.5`B and `0.4926` at `3`B, and the rate of serving the same draw at `n=64` is `0.052`, `0.132`
and `0.228` against `0.016` by chance. The large step is `0.5`B to `1.5`B and the small ones are
above it --- the same shape as the judged gains, arrived at without the instrument whose consistency
this paper distrusts.

**What does not change.** The certificate, as always: `q(y) <= n p_s(y)` holds for any score, so
every one of these 24 cells is certified at `log n` --- `4.159` nats at `n=64` --- whatever the
scorer costs or achieves. Four scorers of one family on one workload cannot locate a capability
threshold in general, and the paper claims none: what it now has is that the threshold is **below**
`1.5`B on this workload, because `1.5`B already saturates.
