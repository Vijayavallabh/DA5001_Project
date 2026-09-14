# Pre-registration: the head-to-head at a third pair

Committed **before the selection arm is generated**. Nothing above the `## Scoring log` line is
edited afterwards.

## Why this arm exists

Section 6's central comparison is a reversal --- the metered decoder gains `+0.072` for `171.3`
nats where selection gains more for `3.175` --- and the obvious objection is that it is one pair.
`results/onset_prediction_frontier_second_pair.md` answered it once, at Llama-3.2-1B, where the
reversal repeated. Two pairs is better than one and is still small, and the paper currently has to
write "the one further pair where the metered decoder runs at all".

A third exists and costs almost nothing, because half of it is already on disk. The metered sweep at
`Llama-3.2-3B-Instruct` against `Llama-3.1-8B-Instruct` was generated for the Proposition 3 breadth
arm (`output/phase5/imit_llama323bi`, `k \in \{-1, 0, 0.5, 1, 3, 20\}`, `500` ordinary prompts).
Only the selection arm at that anchor is missing.

**This pair is the hardest case for selection on record, which is why it is worth running.** Its
imitation rate is `0.3034`, the lowest of the four pairs --- the anchor is *close* to the risky
model, which is exactly the regime where a per-token meter buys the most and where selection, bounded
by its anchor's support, should struggle. At the second pair the metered decoder already "resolves
at three of four budgets, where at the audited pair every budget below `k=10` gains nothing", and
this anchor is closer still.

## What is run

`h1.py` at `--safe-model-path meta-llama/Llama-3.2-3B-Instruct --risky-model-path
meta-llama/Llama-3.1-8B-Instruct --trajectories-per-prompt 8 --k-values 0.0`, the same `500`
ordinary prompts (200 neutral, 150 creative, 150 factual), the same temperature and `200`-token cap
as every other arm. Then `analysis/frontier_pair.py`, which judges both mechanisms **in one pass**
against the same `k=-1` opponent with one shared anchor-alone control, exactly as at the second
pair.

Nothing else changes: the same pointwise reward is not used here, the selector is the same one the
second pair used (`frontier_pair.py`'s own rule), the same two judges, the same opponent.

## Bands, committed before the run

Read on judge B, the registered scorer; judge C reported beside it and never substituted for it.

**T1 -- does the reversal survive a third pair, and the closest anchor yet?** Selection at `n=8`
against the **best** metered arm, the one with the largest gain whatever its budget.

| reading | band |
|---|---|
| REPLICATES | selection's gain CI excludes 0 **and** its gain is within `0.03` of the best metered arm's or higher, at less than a tenth of that arm's realised spend |
| WEAKER | selection's CI excludes 0 but its gain is more than `0.03` below the best metered arm's |
| REVERSED | the metered decoder's gain exceeds selection's by more than `0.03` with non-overlapping intervals |

**WEAKER and REVERSED both cost the paper something, and what they cost is fixed now.** Under
WEAKER, Section 6 must say the reversal holds at two of three pairs and name this one as the
exception, with its imitation rate, because "the anchor is close to the risky model" is a
*prediction* this pre-registration is making and not an excuse to be found afterwards. Under
REVERSED, the `171.3`-against-`3.175` sentence becomes a statement about pairs whose anchor is far
from the risky model, in Section 6 **and** in the abstract.

**T2 -- the cross-pass floor is not crossed by this arm.** Both mechanisms are judged in one pass
against one shared control, so T1 is a within-pass comparison. No number here is compared against
the audited pair's or the second pair's levels; only the *direction* of the comparison is.

**T3 -- what may not be claimed.** `Llama-3.2-3B-Instruct` is not a safe model --- it is trained on
undisclosed data --- so no leakage, certificate, vacuity or `s(x)` number comes from this arm, and
it decodes no protected split at all. Stated so the absence is not later read as a result.

## Excluded alternatives

- Re-running with a different `n`, a different opponent, a different judge or a different anchor if
  T1 reads WEAKER or REVERSED.
- Reporting this pair and dropping either of the two on record.
- Quoting a level from this pass beside a level from another (cautions (e), (m), and the cross-pass
  floor of about `0.04` measured twice in Appendix A).
- Reading the metered decoder's behaviour here as evidence about a *safe* anchor. It is not one.

## Scoring log

## Scoring, 2026-09-14

Run: `scripts/run_frontier_third.sh`, GPU 0, `[f3] generate exit=0`, `[f3] judge exit=0 at 09:08`.
Output `results/frontier_pair_llama323bi.csv`. One pass, one shared control, $500$ prompts per arm.

### T1 --- REPLICATES, and at the pair that was registered as the hardest case

Read on judge B, `Phi-3.5-mini-instruct`, the registered scorer.

| arm | spend, nats | gain over the shared control |
|---|---|---|
| `metered, k=0.5` | `54.848` | `-0.023 [-0.066, +0.021]` |
| `metered, k=1` | `59.254` | `-0.038 [-0.079, +0.001]` |
| `metered, k=3` | `60.279` | `-0.022 [-0.058, +0.014]` |
| `metered, k=20` | `60.634` | `-0.019 [-0.059, +0.021]` |
| **`selection, n=8`** | **`1.204`** | **`+0.150 [+0.106, +0.194]`** |

Selection's interval excludes zero, its gain is above the best metered arm's rather than within
`0.03` of it, and it spends `1/50` of that arm's realised nats. **REPLICATES.**

Judge C, `Meta-Llama-3.1-8B-Instruct`, reported beside it and never substituted for it, agrees and
by more: selection `+0.273 [+0.227, +0.323]` against a best metered arm of `-0.009 [-0.066,
+0.044]` at `59.254` nats, a `49.2x` spend ratio.

### What is new at this pair, and it is the dichotomy's second horn

At the two earlier pairs the metered decoder bought *something* and selection bought more for less.
Here **the metered decoder buys nothing at any budget**: all four gains are negative and all four
intervals contain zero, on both judges. Forty times the published cap --- `k=0.5` to `k=20` ---
moves the realised spend from `54.848` to `60.634` nats, `10.5\%`, and moves the judged gain by
less than its own interval width. This is Proposition~\ref{prop:sparse}'s trivial horn as a
measurement rather than an inference: the anchor is close enough to the risky model that the
decoder is the anchor almost everywhere, and being the anchor almost everywhere is worth nothing
over being the anchor.

It is also what this pre-registration predicted would make the reversal *hard*, and the prediction
was wrong about the direction of the difficulty. A close anchor does not narrow the gap by lifting
the metered decoder; it widens it by flattening it.

### T2 --- the cross-pass floor is not crossed

Every number above is within one judging pass against one shared control, and the two nominally
identical controls --- the metered run's `k=0` arm and the selection run's `n=1` arm, the same
thing generated twice --- are reported rather than smoothed: `-0.009 [-0.053, +0.037]` on judge B
and `+0.010 [-0.044, +0.062]` on judge C. Neither excludes zero, which is the same reading this
paper's generation-run noise floor has at the second pair. No level here is quoted beside a level
from another pass.

### T3 --- what is not claimed

`Llama-3.2-3B-Instruct` is trained on undisclosed data and is not a safe model. No leakage,
certificate, vacuity or `s(x)` number comes from this arm, and it decoded no protected split at
all. The `60.634` nats are a realised spend at this pair and are not comparable to the audited
pair's `171.3` across passes; only the *direction* of the comparison is carried across.

### Manuscript consequence

Section 6's "At the one further pair where the metered decoder runs at all ... the reversal
repeats" becomes two further pairs, and the new pair's flat metered arm is reported as the trivial
horn rather than as a stronger version of the same reversal. Appendix~A gains the third pair's
table beside the second's.
