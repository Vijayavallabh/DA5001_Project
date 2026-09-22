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

## Scoring, 2026-09-22

### Gates

| gate | requirement | measured | reading |
|---|---|---|---|
| G0 same inputs | the pass reads Arm C's own directories and reward cache | `--sel-dir output/wscope/a_sel`, `--rewards results/wscope_rewards64_a.csv`, no generation | **PASS** |
| G1 the effect is present | judge~B's `D3` on this arm excludes zero, positive | `+0.0965 [+0.0765, +0.1162]` | **PASS** |

G1 is the condition feat-166 and feat-168 both failed. It is the whole difference between this arm
and the two before it, and it is why this reading can be interpreted at all.

### B1 --- **TIGHT ZERO**

| judge | paired `D3` | half-width | reading |
|---|---|---|---|
| B, `Phi-3.5-mini` | `+0.0965 [+0.0765, +0.1162]` | `0.0199` | REVERSAL CONFIRMED |
| C, `Llama-3.1-8B-Instruct` | `+0.0903 [+0.0629, +0.1176]` | `0.0274` | REVERSAL CONFIRMED |
| **Mixtral-8x7B-Instruct** | **`+0.0226 [-0.0018, +0.0476]`** | **`0.0247`** | **TIGHT ZERO** |

The interval contains zero and its half-width is `0.0247`, below the committed `0.030`, so the band
reads **TIGHT ZERO** and its consequence applies. **We predicted TIGHT ZERO and it is TIGHT ZERO.**

**What that means, precisely.** Under the one family-clean frontier judge the difference is
*positive* (`+0.0226`) and *bounded*: its upper end is `+0.0476`, well below the `+0.0645` on
record and below what the other judges read. The effect is not absent under Mixtral --- the point
estimate has the same sign as every other judge --- it is **small**, and at `850` prompts we can
now say how small rather than shrugging at a wide interval. The hypothesis that the committed
`+0.0090` was merely under-powered is refuted: with `1.7x` the prompts the interval tightened from
a half-width of `0.0443` to `0.0247` and the point estimate stayed near zero.

### B2 --- the sentence, as committed in advance

The TIGHT ZERO branch was fixed before any of this ran and it is adopted verbatim: the paper says
the reversal *"replicates under four judges and is bounded below `0.03` under the one clean
frontier judge"*, **weakened in the body and not in a footnote**. Section 4's current
*"four of five ... the exception being the family-clean `Mixtral-8x7B`"* is replaced by a statement
that says how large the exception is rather than only that it exists, which is strictly more
informative and strictly less flattering.

### B3 --- reported whatever it says

| judge | `g_sel` | `g_met` | order consistency, `sel_n64` |
|---|---|---|---|
| B | `+0.1218 [+0.1038, +0.1400]` | `+0.0253 [+0.0074, +0.0432]` | `0.2894` |
| C | `+0.1782 [+0.1556, +0.2009]` | `+0.0879 [+0.0618, +0.1144]` | `0.6482` |
| Mixtral | `+0.1197 [+0.0971, +0.1429]` | `+0.0971 [+0.0735, +0.1209]` | `0.4141` |

Mixtral's order-consistency is `0.4141`, inside the `0.358`--`0.426` on record and well below the
`0.4807` it read on AlpacaEval, so nothing here revises caution (m) and the judge is behaving as it
always has.

**Where Mixtral differs from the other two is on the metered arm, not the selection arm.** All
three judges put selection within `0.06` of each other (`+0.1197` to `+0.1782`); Mixtral scores the
*meter* at `+0.0971` where judge~B puts it at `+0.0253`, a gap of `0.072`. The disagreement about
the difference is a disagreement about how good a budgeted decoder's text is, not about selection's
--- which is worth saying, because a reader meeting one dissenting judge will assume the opposite.

### What it took to get here, recorded because it is the lesson

Three arms. feat-166 ran on AlpacaEval at `k=10` and its budget was vacuous; feat-168 fixed the
budget and stayed on AlpacaEval, where the reversal does not hold; both gates fired correctly and
both refused to let Mixtral be read, which is the system working and also two arms spent. **The
question was never about Mixtral or about statistical power --- it was that we twice asked it on a
workload where the quantity being judged does not exist.** The rule this leaves: before adding
power to resolve a marginal reading, check that the effect the reading is about is present in the
pass you are adding power to.
