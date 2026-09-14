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
