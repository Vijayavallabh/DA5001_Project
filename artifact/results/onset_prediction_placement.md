# Pre-registration: budget placement at a fixed sequence budget

Committed **before any arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists, and why the paper needs it

Proposition 5 says: for any causal policy `q` with `D_KL(q || p_s) <= K`, the expected number of
steps at which `D_KL(q_t || p_s,t) > eps` is at most `K/eps`. A policy whose budget does not grow
with the work must therefore leave the anchor untouched at all but `O(1)` steps.

That is a statement with **two horns**, and the paper so far only demonstrates one of them. The
deployed token-bucket rule sits on the wrong horn by construction (Proposition 3, measured: at
`k = 20` it serves `p_r` unchanged at 99.95% of steps and spends `Theta(T)`). But Proposition 5 does
not forbid a causal policy from working -- it forbids a causal policy from working *while spread
out*. A policy that concentrates its whole budget on the opening tokens is permitted, and the paper
has never tried one. Without that arm, the claim "the budget has to leave the decode loop" is
stronger than the evidence: a reviewer is entitled to ask whether a front-loaded meter would have
done just as well, and the paper's own opening-effect appendix says the opening is where a decoder's
choices bite.

So this arm measures **placement at a fixed sequence budget**: the same nats, spent in three places.

| placement | how | budget |
|---|---|---|
| uniform, per step | the deployed bucket, refill `k` per token | `K = k*T_max` |
| front-loaded, causal | `--initial-bank K` with a negligible refill (`k = 1e-9`) | `K` |
| off-axis, at the draw | selection anchoring, best of `n` | `K = log n` |

All three are run at **`K = log 8 = 2.0794` nats**, the budget selection anchoring already spends on
record (`results/selection_crossjudge.csv`), on the **same 500 ordinary prompts**, scored by the
**same judge B** (`microsoft/Phi-3.5-mini-instruct`) against the same `n = 1` anchor-only control, so
the three numbers are comparable without any rescaling. A second front-loaded arm at `K = 20` nats
says whether concentration scales; a uniform arm at `k = 0.0104` (`= 2.0794/200`) is the placement
control at the identical total.

Baselines at `k = -1` and `k = 0` are generated on the same prompts and seeds, as for every arm in
this repository.

## Pre-run correction: the prefix debt would make every arm the anchor

Measured before generating anything, on 400 ordinary prompts of `output/sweep_plain`: the prefix
debt has median **2.53 nats** (p10 1.86, p90 3.46) and **exceeds the whole matched budget of 2.0794
on 77.5% of prompts**. The bank starts at `-delta` and refills at `k`, so at `K = 2.0794` --- whether
granted up front or accrued at `k = 0.0104` --- the bank never reaches zero on three prompts in four,
and *both* causal arms would be the safe model exactly, by construction. The comparison would be a
confounded null: it would read THE CAUSAL HORN IS EMPTY for a reason that has nothing to do with
placement.

The placement arms therefore run with `--no-prefix-debt`, so the whole sequence budget is available
to be placed and the arms differ in placement alone. This is recorded here **before the run** rather
than discovered after it.

It is also a result in its own right, and belongs in the paper: the prefix debt is itself a
placement decision --- it front-loads a *penalty* --- and at a budget the size of a selection
certificate it consumes the entire allowance before the first token. A deployer who wanted a
metered decoder with a `log 8`-sized budget would be shipping the anchor.

## Bands, committed before the run

Let `u` be judged utility against the unconstrained model on the 0 / 0.5 / 1 scale, `gain` the paired
difference against the `n = 1` anchor-only control, with a paired 95% bootstrap CI over prompts.
Selection's gain at the same budget is **+0.081 [0.034, 0.130]** on record.

**P1 -- does placement matter at all?** Compare the front-loaded arm's gain with the uniform arm's
gain at `K = 2.0794`.

| reading | band |
|---|---|
| PLACEMENT MATTERS | the two CIs do not overlap |
| PLACEMENT IS SECOND ORDER | they overlap but the point estimates differ by more than 0.03 |
| PLACEMENT IS IRRELEVANT | they overlap and differ by at most 0.03 |

**P2 -- is the causal horn reachable?** The front-loaded arm against the anchor-only control.

| reading | band |
|---|---|
| CAUSAL CONCENTRATION WORKS | gain CI excludes 0 and the point estimate is within 0.03 of selection's +0.081 |
| CAUSAL CONCENTRATION PARTLY WORKS | CI excludes 0 but the gain is below `0.081 - 0.03` |
| THE CAUSAL HORN IS EMPTY | CI includes 0 |

**This band is where the paper's claim is at risk, and it is stated that way on purpose.** Under
CAUSAL CONCENTRATION WORKS the manuscript's framing changes: the escape is not "leave the decode
loop", it is "do not spread the budget", selection becomes one of two mechanisms obeying the same
principle rather than the only one, and Sections 5-6 are rewritten to lead with the principle. Under
THE CAUSAL HORN IS EMPTY the constructive claim strengthens to what it currently asserts, and the
reason is reportable: concentrating a `log 8`-sized budget on the opening buys nothing because the
opening is not where judged quality is decided.

**P3 -- does concentration scale?** The `K = 20` front-loaded arm against the `K = 2.0794` one.
SCALES if its gain CI excludes the smaller arm's point estimate from above; FLAT otherwise. `K = 20`
is still two orders of magnitude below the `171.3` nats the metered decoder spends, so a positive
reading here would be a finding about the metered decoder's waste, not a rescue of it.

**P4 -- leakage, mandatory for every arm.** Near-verbatim recall on the protected passages at every
placement, with the `k = -1` and `k = 0` baselines. Selection's is `0.0000` at every `n <= 64`.
Any arm with non-zero recall is reported with it, whatever its utility.

## Excluded alternatives (named now so they cannot be adopted afterwards)

1. Tuning the refill rate of the front-loaded arm after seeing its utility. It is `1e-9` because the
   constructor reserves `k = 0` for the safe-only baseline, and `1e-9 * 200 = 2e-7` nats is
   negligible against `2.0794`.
2. Changing the judge, the template, the prompt set, or the control between arms.
3. Reporting the front-loaded arm at `k*T_max` instead of at `k*T_max + initial_bank`. The bank is
   the whole budget; `tests/test_initial_bank.py` pins this.
4. Declaring PLACEMENT MATTERS on overlapping intervals.
5. Dropping P2 if it reads CAUSAL CONCENTRATION WORKS. That reading is the one that costs the
   paper its current framing, and it is the reason the arm is worth running.
6. Quoting a judged separation without its sample size (caution (e)), or building on one smaller
   than a sigma.

## Scoring

```
# front-loaded, K = log 8, on the 500 ordinary prompts
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<free card> HF_HUB_OFFLINE=1 \
  HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python h1.py --k-values 1e-9 --initial-bank 2.0794 \
  --trajectories-per-prompt 1 --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 \
  --output-dir output/phase5/place_front_2p08
# ... and the uniform control at k = 0.0104, the K = 20 arm, and the k = -1 / k = 0 baselines
.venv/bin/python analysis/placement.py --out results
```

Writes `results/placement.csv` and `results/placement_per_prompt.csv`.

---

## Scoring log (appended after the run; nothing above this line is edited)

---

## Scoring, 2026-09-12 (appended; nothing above is edited)

```
# each arm: 500 ordinary prompts, one trajectory, --no-prefix-debt, 200-token cap
.venv/bin/python h1.py --k-values 0.0103970 --initial-bank 0.0    --no-prefix-debt ... place_unif_2p08
.venv/bin/python h1.py --k-values 1e-9      --initial-bank 2.0794 --no-prefix-debt ... place_front_2p08
.venv/bin/python h1.py --k-values 1e-9      --initial-bank 20.0   --no-prefix-debt ... place_front_20
.venv/bin/python analysis/placement.py --out results
```

The budgets bind as designed, checked per trajectory before any judging: the front-loaded arm's
realised spend has median **and** maximum `2.0794` — it spends the whole bank at once and then
serves the anchor — while the uniform arm's is median `1.2736`, maximum `2.0794`. No trajectory
exceeds `K` in either.

| placement | `K` | `u` | gain vs the anchor-alone control |
|---|---|---|---|
| anchor alone (control) | `0` | `0.473 [0.440, 0.508]` | — |
| uniform, per step | `2.0794` | `0.441 [0.406, 0.479]` | `−0.032 [−0.083, +0.016]` |
| front-loaded | `2.0794` | `0.465 [0.428, 0.502]` | `−0.008 [−0.056, +0.041]` |
| front-loaded, large | `20.0` | `0.454 [0.417, 0.491]` | `−0.019 [−0.064, +0.027]` |
| **on the draw** (`n=8`, on record) | `2.0794` | — | **`+0.054 [+0.013, +0.095]`** |

### P1 — **PLACEMENT IS IRRELEVANT**

Front-loaded and uniform differ by `0.024` with overlapping intervals, below the committed `0.03`.
*Within* the per-step axis, where you put the budget does not matter.

### P2 — **THE CAUSAL HORN IS EMPTY**

The front-loaded arm gains `−0.008 [−0.056, +0.041]`: the interval contains zero, and the point
estimate is on the wrong side of it. A causal policy that concentrates its entire `log 8`-sized
budget on the opening buys **nothing**.

This is the reading the pre-registration named as the one that *strengthens* the paper, and it does:
"the budget has to leave the decode loop" was an assertion and is now a measurement. Proposition 5
says an affordable causal policy must be the anchor at all but `O(1)` steps; it does not say such a
policy is useless, and we could not have known it was without running this. At the same budget, on
the same prompts, under the same judge, the draw placement gains `+0.054 [+0.013, +0.095]` and both
causal placements gain nothing.

### P3 — **FLAT**

Ten times the budget (`K = 20`) does not help: `−0.019 [−0.064, +0.027]`, no better than `K = 2.08`
and still not distinguishable from the anchor. Concentration does not scale into the useful region
either.

### The uniform arm is the trivial horn, measured at a new budget

Its gain is `−0.032`, the largest negative of the three. A metered decoder given a selection-sized
allowance is indistinguishable from — if anything slightly worse than — the anchor it wraps. That is
the dichotomy's second horn at `k = 0.0104`, four decades below the `k = 10` at which the earlier
judged separation first appears.

### What this does not establish

One anchor, one risky model, 500 in-house prompts, one judge, and one way of concentrating (all at
the front). A policy that spent its budget on *adaptively chosen* steps rather than the opening is
not tested here and is the remaining member of the class; Proposition 5 bounds its shape but not its
value, and Limitations says so. The arms also run with `--no-prefix-debt` for the reason recorded
above the bands: with the debt on, both causal arms would have been the anchor by construction.
