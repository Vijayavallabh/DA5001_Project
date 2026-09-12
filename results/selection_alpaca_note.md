# Selection anchoring on AlpacaEval-805 — an unregistered extension, reported as one

> **Correction, 2026-09-12 (see the last section).** The section below headed "This is the
> support ceiling, measured with a standard instrument" is **refuted** by
> `results/onset_prediction_domain_breadth.md`. The support-ceiling *explanation* of the
> weakening has no evidence at this scale. The measured result above it stands unchanged.

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

---

## Correction, 2026-09-12: the support-ceiling explanation is refuted

The section "This is the support ceiling, measured with a standard instrument" asserted that the
weaker AlpacaEval gain is the anchor's support ceiling showing through, and closed by saying "the
manuscript should say so in those terms". **The manuscript must not.** That explanation was tested
against a pre-registered split and does not survive it.

`analysis/domain_breadth.py`, registered in `results/onset_prediction_domain_breadth.md` before any
per-domain number was computed, cut the 805 AlpacaEval prompts into the five sources the benchmark
ships and MT-Bench into two committed families, and asked whether the gain tracks how well the
anchor can already do the task. It does not, in either judge:

* **D1 is uninformative.** The literal Spearman is `-0.786` (judge C) and `-0.703` (judge B) --- the
  *opposite* of the ceiling prediction --- but that is the direction the pre-registration named as
  mechanically cheap, and a within-prompt exchangeability null in which selection does nothing
  already yields `-0.365 +/- 0.342` and `-0.339 +/- 0.350`. The observed values sit at
  `P = 0.0947` and `0.1651` against that null. Seven cells cannot separate the two.
* **D3 goes the wrong way and has no artefact in it,** because it compares control levels only:
  MT-Bench's open-ended family scores *below* its constrained family in both judges (`0.200` vs
  `0.287`, `0.300` vs `0.425`).

What is refuted is the empirical claim. The theorem behind it --- `q(y) <= n p_s(y)`, so a served
string must be one the anchor would have drawn --- is untouched and stays in Limitations, where it
is a statement about what the mechanism can never do rather than an explanation of a number.

So the paper's line on breadth is: **the gain is weaker off the in-house prompt set, one judge does
not resolve it on AlpacaEval, MT-Bench at 80 prompts resolves nothing in either direction, and we
cannot say why.** The Comma-7B arm named under "What would settle it" is still the right next test
and is still queued; its bands are committed. Until it lands, "anchor-bound" is a hypothesis and is
labelled as one.
