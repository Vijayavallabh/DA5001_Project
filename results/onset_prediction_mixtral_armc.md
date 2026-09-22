# Pre-registration: Mixtral, at last, on a workload where the comparison exists (feat-171)

Committed **before any judging runs**. Nothing above the `## Scoring log` line is edited afterwards.

## Why

Section 4's one UNRESOLVED reading is `Mixtral-8x7B-Instruct`, the only family-clean frontier
judge, at `+0.0090 [-0.0355, +0.0530]` on `500` prompts. Two attempts to settle it have failed and
**both failed for the same reason, which we only understood this morning**: feat-166 and feat-168
both ran on AlpacaEval, where the headline reversal does not hold at all, so judge~B read the
difference *negative* and their own gates correctly refused to let Mixtral be read. The question
was never about Mixtral; the workload was wrong.

feat-170 Arm C is the workload where the comparison exists: our own `850` ordinary prompts, a
budget that genuinely binds (`8.86%` of steps, `2.0%` of completions shared with the opponent),
and judge~B reading `+0.0965 [+0.0765, +0.1162]`, REVERSAL CONFIRMED. Its generations and its
`54{,}400`-candidate reward cache are on disk. **Only the judge changes.**

`1.7x` the prompts of the reading on record, on a workload where the effect is present.

## What runs

`analysis/order_averaged_h2h.py` over Arm C's existing directories --- no generation, no
re-scoring --- under two judges:

- **`mistralai/Mixtral-8x7B-Instruct-v0.1`**, `--device-map auto` across two cards. The question.
- **`meta-llama/Meta-Llama-3.1-8B-Instruct`** (judge~C), one card. A second reading on the same
  text, and the paper already discloses that judge~C is the same checkpoint as the risky model
  every arm is judged against, a self-preference risk (caution (aa)); it is reported with that
  caveat attached, as it always is.

## Gates

- **G0 (the pass reproduces itself).** Judge~B's `D1`, `D2` and `D3` on Arm C are fixed and on
  record: `+0.1218`, `+0.0253`, `+0.0965`. The new passes must reproduce **the selection arm's own
  rank-`0` text** exactly --- they read the same directories, so this is an identity check on the
  inputs, not a re-measurement.
- **G1 (the workload is the one where the effect is present).** Judge~B's `D3` on this arm excludes
  zero on the positive side. Already true and already committed; restated here because it is the
  condition feat-166 and feat-168 both failed, and reading Mixtral without it is what produced two
  wasted arms.

## Bands, carried over verbatim from feat-166

| reading | band |
|---|---|
| **REVERSAL HOLDS** | Mixtral's paired `D3` `> 0` and the interval excludes zero |
| **TIGHT ZERO** | interval contains zero **and** half-width `< 0.030` |
| **STILL UNRESOLVED** | interval contains zero with half-width `>= 0.030` |
| **INVERTS** | `< 0` and the interval excludes zero |

**B2 --- the paper's sentence, fixed per branch.** REVERSAL HOLDS: the exception is retired and
Section 4 says five of five, naming the sample size and the workload that settled it. TIGHT ZERO:
*"replicates under four judges and is bounded below `0.03` under the one clean frontier judge"* ---
weakened in the body, not in a footnote. STILL UNRESOLVED: the current wording stands and this arm
is reported as a third failure to settle it. INVERTS: the headline is withdrawn as judge-dependent.

**B3 --- reported whatever it says:** both judges' paired `D1`, `D2`, `D3`, and Mixtral's
order-consistency against the `0.358`--`0.426` on record and the `0.4807` it read on AlpacaEval.

## Excluded in advance

- Quoting any judged LEVEL from this pass beside a level from another (caution (ap)); every reading
  is a paired difference within this pass.
- Pooling with feat-166, feat-168, or the committed `500`-prompt pass.
- Dropping judge~C if it disagrees with Mixtral, or Mixtral if it reads INVERTS.
- Re-running the anchor draws, the opponent or the reward cache. They do not depend on the judge,
  and re-drawing them would change two things at once (caution (v)).

## What we predict

**TIGHT ZERO**, unchanged from feat-166, and for the same reason: `+0.0090` sits `0.20` interval
half-widths from zero, which caution (ap) says is marginal in the sense that survives re-drawing
only as marginal. We note the standing risk that at `850` prompts the interval may still be too
wide to distinguish TIGHT ZERO from STILL UNRESOLVED, in which case the honest report is the
latter and we say the question is not settleable with the prompts we have.

## Compute

Host B, judging only. Mixtral about `6` hours across two cards; judge~C about an hour on one.

## Scoring log
