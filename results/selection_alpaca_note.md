# Selection anchoring on AlpacaEval-805 — an unregistered extension, reported as one

**This arm has no committed bands.** Every other arm in this repository was pre-registered with a
refuting band before it ran; this one was not, because the corpus and the run were built in the same
pass on 2026-09-12. That is a deviation from the project's discipline and it is recorded here rather
than repaired retroactively: no band is invented after the fact, and nothing below may be written up
as a confirmed prediction.

## What was run

`data/bench/alpaca/` — the full AlpacaEval instruction set, 805 prompts, through the **factual**
slot so each instruction reaches the model verbatim. Eight anchor samples per prompt from
TinyComma-1.8B, the identical pointwise reward and two-judge protocol as feat-088, judged against
the unconstrained Llama-3.1-8B-Instruct on the same prompt, 400-token cap.

## Result

| judge | `n=1` | `n=8` | gain | Spearman(`u`, `log n`) |
|---|---|---|---|---|
| B (Phi-3.5-mini) | `0.3335` | `0.3646` | `+0.0311 [−0.0006, +0.0621]` | `+0.40` |
| C (Meta-Llama-3.1-8B-Instruct) | `0.2267` | `0.2932` | `+0.0665 [+0.0342, +0.1006]` | `+1.00` |

Against the in-house 500, where the same protocol gave `+0.054 [+0.013, +0.095]` on judge B and
`+0.073 [+0.027, +0.120]` on judge C.

## Reading it honestly

**The gain is weaker here, and on one judge it is not resolved.** Judge B's interval includes zero
by `0.0006` and its sweep is not monotone (`ρ = +0.40` against `+0.99` in-house). Judge C's gain
replicates at a comparable size with a monotone sweep. One judge agreeing and one not resolving, at
805 prompts, is a weaker result than the in-house arms and must be reported as one.

**The absolute levels are not AlpacaEval win-rates and must never be quoted as such.** The anchor is
a 1.8B *base* model answering instructions against an *instruction-tuned* 8B. Much of the gap at
`n=1` (`0.33` and `0.23`, against `0.44` and `0.43` on the in-house completion prompts) is format
rather than capability. The within-anchor *gain* is unaffected by that, which is why the gain is the
quantity reported.

**This is the support ceiling, measured with a standard instrument.** Limitations already says a
selection mechanism cannot exceed what its anchor can write. The prediction that follows is exactly
what is seen: move from completion-style prompts, which a base anchor can do, to instruction
following, which it largely cannot, and the headroom selection has to work with shrinks. That is a
confirmation of a stated limitation, not a refutation of the mechanism — but it is also the clearest
evidence yet that the constructive claim is anchor-bound, and the manuscript should say so in those
terms rather than lead with `+0.142`.

## What would settle it

The same arm at **Comma-7B**, already queued: a stronger, openly licensed base anchor with far more
support on instruction-style prompts. If the gain recovers there, the reading is the support ceiling;
if it does not, the reading is that the in-house prompt set flatters the mechanism. Either way the
answer belongs in the paper, and the band for that one is committed in
`results/onset_prediction_selection_breadth.md`.
