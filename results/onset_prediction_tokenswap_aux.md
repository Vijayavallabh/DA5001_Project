# Pre-registration: TokenSwap's auxiliary is a stated requirement, not a free parameter (feat-167)

Committed **before any rung is run**. Nothing above the `## Scoring log` line is edited afterwards.

## Why

feat-165 runs TokenSwap with `TinyComma-1.8B`, chosen because it is already vetted at `0.000` and
shares Llama-3's tokenizer. It is also roughly `22x` larger than the `DistilGPT-2` their paper
uses, so **every reading feat-165 produces is at an auxiliary more capable than the method asks
for**. That cuts two ways and the arm cannot tell them apart: a bigger auxiliary should cost less
utility, and it may also memorise more.

This is the same question this paper already asked about its own scorer and answered with a ladder
(Appendix~A.1, `0.494`B to `72.7`B). The mechanism's requirement is a cost, and a requirement whose
size is unmeasured is a hidden parameter.

**And it settles, for their own configuration, the thing Appendix~J said had to come first.** The
sentence was: the auxiliary's *"own contamination would have to be vetted before any leakage number
from it meant anything"*. `DistilGPT-2` descends from GPT-2, trained on WebText, which is scraped
from outbound Reddit links and is not an openly licensed corpus. Nobody has run our vetting check
on it. We will.

## What runs

**Half A --- vetting, four auxiliaries, the protocol Appendix~D already uses for anchors.**
Near-verbatim recall of `50`-token windows on the `100` protected passages at a `100`-token raw
prefix, each model alone, no defence. Candidates: `DistilGPT-2` (`82`M, theirs), `KL3M-170m`,
`Pleias-350m-Preview`, `TinyComma-1.8B` (the rung feat-165 used, re-run here so the ladder is one
pass).

**Half B --- the utility rung at each auxiliary.** `analysis/tokenswap_decode.py --arms tokenswap`
on the same `850` ordinary prompts with `--chat`, the flags `scripts/run_memfree.sh` uses. The
rule-off control is not re-run: it does not depend on the auxiliary and feat-165 generates it once.
`G` is paired across vocabularies where they differ, by the mapping their paper relies on --- a
surface form counts only if it is a single token in **both**, and the count that survives is
reported per rung.

## Gates

- **G0 (the pairing survives).** At most `20` of the `110` words may fail to pair into a given
  auxiliary's vocabulary. Above that the rule being run is materially weaker than the one the
  authors specify and that rung is reported as NOT RUN rather than as a weak result.
- **G1 (the rule binds at every rung).** `changed_frac > 0.01`, as in feat-165.

## Bands

- **B1 --- does their own auxiliary pass our vetting check?** Near-verbatim recall of `DistilGPT-2`
  alone on the `100` passages. **CLEAN** at `<= 0.01`, **LEAKS** above it. The same reading is
  reported for all four.
- **B2 --- does suppression depend on the auxiliary?** feat-165 reads `0.0000` recall and `0`/`100`
  at ROUGE-L `>= 0.5` with the `1.8`B rung. Any rung that reads above `0.05` recall makes
  suppression auxiliary-dependent, which the paper would then have to state.
- **B3 --- does utility?** The judged order-averaged gain at each rung against the shared rule-off
  control. Reported as a series with its shape named, the way the scorer ladder is
  (MONOTONE / NOT MONOTONE, and where the peak is).

## Excluded in advance

- Dropping a rung because it is unflattering to TokenSwap or to us.
- Treating a `DistilGPT-2` leakage reading as a criticism of their *method*: it is a property of
  the auxiliary, and the method's own text says the auxiliary is a choice. What it would criticise
  is running the method without checking that choice --- which is what our own Appendix~J said.
- Re-running the control per rung and comparing rungs against different controls.

## What we predict

`DistilGPT-2` **LEAKS** is genuinely uncertain: at `82`M it is small enough that Appendix~D's own
finding --- that small openly licensed anchors read `0.000` --- may hold for it too, and its
training corpus is the only reason to doubt it. We predict **CLEAN**, at `<= 0.01`, and note that
predicting the comfortable outcome is exactly the case where a committed band matters. On B2 we
predict suppression holds at every rung, since a smaller auxiliary memorises less. On B3 we
predict the gain falls as the auxiliary shrinks, and that the `82`M--`350`M rungs cost more than
the `1.8`B one.

## Compute

Host B, four cards. Half A about `40` minutes. Half B about `90` minutes per rung; the
cross-tokenizer rungs are slower because the auxiliary must be re-encoded from text at every step.

## Scoring log
