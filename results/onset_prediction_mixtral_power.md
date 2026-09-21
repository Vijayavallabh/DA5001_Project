# Pre-registration: is Mixtral's exception a small effect or a wide interval? (feat-166)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists

Section 4 says the headline reversal *"replicates across judges, not across opponents"*: five
judges, `3.8`--`72`B in four families, re-scored byte-identical text, and **four of five** put the
paired difference clear of zero. The exception is the only family-clean frontier judge,
`Mixtral-8x7B-Instruct`, at **`+0.0090` `[-0.0355, +0.0530]`**.

That reading is UNRESOLVED in the registered sense --- the interval contains zero --- and the paper
reports it as such. But UNRESOLVED has two very different causes and the paper cannot currently
tell them apart. Either the effect under that judge is genuinely near zero, or `500` prompts is too
few to see a `+0.06` effect through that judge's noise. **Nothing about the judge can settle it:
`judge_batch` is greedy, so re-judging the same text gives the same answer exactly, and
order-averaging already removes the position lottery.** The only lever is more prompts.

This matters more than any other open reading in the paper. Every other judge agrees; if the one
clean frontier judge is measured at a tight zero, the honest summary of the headline changes from
"replicates across judges" to "replicates across judges except the largest clean one", and the
paper must say so.

## What runs

A **new, self-contained pass on a fresh prompt set**: `AlpacaEval-805`
(`data/bench/alpaca`, the factual slot so no `Complete the prefix:` header is prepended), which
feat-096 already established carries **no prompt-set effect** (A2). `805` prompts against the
committed `500` is a `1.61x` larger sample and a standard public benchmark rather than our own.

Nothing is compared across passes. Caution (ap) forbids setting a judged *level* from one sweep
against one from another, and this arm never does: every reading below is a **paired difference
within this pass**, and the committed `+0.0090` is quoted only as the prior it was registered
against.

| card | cell |
|---|---|
| 3 | anchor draws, `k=0`, `--trajectories-per-prompt 64` |
| 4 | metered decoder, `k=10`, one completion |
| 5 | the fixed opponent, unconstrained `Llama-3.1-8B-Instruct` |

Then `Qwen2.5-7B-Instruct` scores all `51{,}520` candidates once, and
`analysis/order_averaged_h2h.py` judges four arms in both presentation orders under **two** judges
--- judge B `Phi-3.5-mini` (the one on record) and `Mixtral-8x7B-Instruct` (the exception). The
other three judges are not re-run: they already agree, and spending four days re-confirming them
would buy nothing this arm is about.

## Gates, read before any band

- **G0 (the pipeline reproduces itself).** Judge B's paired difference on this pass must be
  positive with its interval clear of zero. Judge B is the arm on record at `+0.0645`
  `[+0.030, +0.0995]`; if it cannot reproduce the direction on a fresh prompt set, the pass is
  measuring something else and Mixtral's reading here is uninterpretable. **This is a direction
  gate, not a magnitude gate** --- a different prompt set may legitimately move the size.
- **G1 (the sample is bigger).** Judge B's interval half-width here must be **below** the `0.0348`
  on record. If a `1.61x` larger sample does not narrow the interval, the extra prompts are not
  buying power and B1 cannot be read as resolving anything.

## Bands, committed before the data

**B1 --- what Mixtral says with more prompts.**

| reading | band |
|---|---|
| **REVERSAL HOLDS** | difference `> 0` and interval excludes zero |
| **TIGHT ZERO** | interval contains zero **and** its half-width is `< 0.030`, i.e. the effect is bounded below the `+0.0645` on record |
| **STILL UNRESOLVED** | interval contains zero with half-width `>= 0.030` |
| **INVERTS** | difference `< 0` and interval excludes zero |

**B2 --- the paper's sentence, fixed in advance for each branch.**
REVERSAL HOLDS: the exception is retired and Section 4 says five of five, naming the sample size
that settled it. TIGHT ZERO: *"replicates under four judges and is bounded below `0.03` under the
one clean frontier judge"* --- the claim is weakened in the body, not in a footnote. STILL
UNRESOLVED: the current wording stands and this arm is reported as having failed to settle it.
INVERTS: the headline is withdrawn as judge-dependent.

**B3 --- reported whatever it says:** the paired `g_sel` and `g_met` under both judges, and
Mixtral's order-consistency rate on this pass against the `0.358`--`0.426` on record.

## Excluded in advance

- Pooling this pass with the committed one. The prompt sets are disjoint and nothing about pooling
  was registered; averaging them would be the post-hoc rescue this document exists to exclude.
- Quoting any level from this pass beside a level from another (caution (ap)).
- Adding judges after seeing B1, or dropping Mixtral if it reads INVERTS.
- Reading B1 if G0 fails.

## What we predict

**TIGHT ZERO.** The committed reading's point estimate is `+0.0090`, which is `0.20` interval
half-widths from zero --- caution (ap)'s rule says a reading that marginal does not survive a fresh
draw as anything but marginal, and the effect under this judge looks genuinely small rather than
merely noisy. If that is right the paper gets weaker and more accurate in the same edit.

## Compute

Host B. Generation about `3` hours across three cards, rewards about `10` minutes, judge B about
an hour, Mixtral on two cards about `6` hours.

## Scoring log

## Scoring, 2026-09-22

### G0 --- FAILED. B1 is not read.

| gate | requirement | measured | reading |
|---|---|---|---|
| G0 direction | judge B's paired `D3 > 0`, interval clear of zero | `-0.0957 [-0.1124, -0.0792]` | **FAIL** |
| G1 power | half-width below `0.0348` | `0.0166` | pass |

The registration excludes *"Reading B1 if G0 fails"* and that exclusion is honoured: **no branch of
B1 is claimed, no sentence from B2 is adopted, and Mixtral's reading on this pass is not quoted as
resolving anything.** Section 4's current wording stands untouched.

G1 passed --- `805` prompts really did buy power, the interval is `2.1x` tighter --- which makes the
failure sharper rather than softer. The pass is not underpowered. It is measuring the wrong thing,
and G0 exists to say so before any band is read.

### Why G0 failed: the arm labelled "metered decoder" is not one

G0's registered purpose is *"if it cannot reproduce the direction on a fresh prompt set, the pass is
measuring something else"*. It is, and the cause is structural and measured, not inferred.

| | committed pass (our prompt set) | this pass (AlpacaEval-805) |
|---|---|---|
| steps where the `k=10` budget is active | `261,239` of `3,118,893` --- **`8.376%`** | `26` of `160,227` --- **`0.016%`** |
| metered completions byte-identical to the unconstrained opponent | **`0`/`500`** | **`794`/`805`** |
| `D2` metered gain over the anchor | `+0.0390` | `+0.1792` |
| metered arm's order consistency | `0.346` (UNUSABLE) | `0.868` (**STABLE**) |

On AlpacaEval at `k=10` the constraint is active on one step in six thousand, a factor of `520`
below the same budget on our own corpus, and `98.6%` of the served completions are the
unconstrained risky model **byte for byte**. There is no budgeted decoder in this pass. `D2`
therefore measures *the unconstrained `Llama-3.1-8B-Instruct` against a `1.8`B base-model anchor on
an instruction benchmark*, which it wins easily and consistently --- hence `+0.1792` and the
`0.868` consistency, the only arm in any pass this project has run that the judge can tell apart
reliably. And `D3` measures *selection from that anchor against the unconstrained risky model*,
which is not the comparison the reversal is about and was never a claim this paper makes.

**Per caution (w) the arm is INVALID rather than FAILED.** The defect is in our specification: the
"What runs" table named `k=10` on a corpus where `k=10` is vacuous, and nothing in the
pre-registration checked that the budget binds there. A question must not be retired by our own
defect, so Mixtral's `+0.0090 [-0.0355, +0.0530]` remains exactly as unresolved as it was.

### The premise that licensed the prompt set was too broad

The registration justifies AlpacaEval by saying feat-096 *"already established carries **no
prompt-set effect** (A2)"*. A2 is real, and it is about a different quantity: it compared the
**selection gain across anchors** on AlpacaEval and found the prompt set was not flattering the
mechanism. It says nothing about whether a **per-token budget binds** on that corpus, which is the
property this arm's metered cell depends on. A finding about one quantity was used to license a
measurement of another, and the two share only the corpus. Caution (v)'s rule --- a reference
number carries its protocol --- extends to a reference *finding*: **a prior result licenses a new
arm only for the quantity it was measured on.**

### What this pass does establish, reported because it was measured

It is a clean, independent instance of the paper's own dichotomy, on a corpus we did not choose and
a benchmark we do not control. `K = kT` at `k=10` over `200` tokens is `2{,}000` nats, and on
AlpacaEval the decoder never comes near it: the certificate is formally intact and operationally
**vacuous**, because the object it certifies is the risky model itself. That is the vacuous horn,
measured at `0.016%` activity, and it is the first time this project has caught it on a standard
public benchmark rather than by construction. The number is reported here; it is **not** promoted
into the manuscript from an INVALID arm, and it will be re-derived by feat-168 in a pass built to
measure it.

### B3 --- reported whatever it says, and it says the arms are not comparable

Under judge B: `g_sel = +0.0835 [+0.0665, +0.1016]`, `g_met = +0.1792 [+0.1630, +0.1950]`.
Under Mixtral: `g_sel = +0.0590 [+0.0376, +0.0814]`, `g_met = +0.2165 [+0.1960, +0.2363]`.
Mixtral's order-consistency on this pass is `0.4807` at `sel_n64` and `0.5230` at the anchor,
against `0.358`--`0.426` on record --- higher, but still in the range the paper calls unusable, so
nothing here revises caution (m). Both judges agree with each other on this pass, and both are
answering the question the pass actually posed rather than the one it was registered to pose.

### What happens next

The question is unchanged and the instrument has to be rebuilt: the metered cell must run at a
budget that **binds on this corpus**, chosen by matching the committed pass's `8.376%` activity
rather than by reusing a `k` whose meaning does not transfer. That rule is fixed in
`results/onset_prediction_mixtral_power_k.md` (feat-168) before the calibration sweep runs, so the
`k` cannot be chosen to suit an answer. Nothing else about the arm changes --- same prompts, same
models, same two judges, same bands.
