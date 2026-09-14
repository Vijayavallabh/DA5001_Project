# Session handoff — 2026-09-14 (morning)

## Current objective

**feat-111** is running: five contaminated anchors, testing Proposition 4 where it can actually fail (`results/onset_prediction_contaminated_anchor.md`). Every other leakage arm has a clean anchor and a base rate of zero, so `n x 0 = 0` holds whatever is measured and the multiplication bound has never been tested. GPUs 0/1/2/4, a fifth queued behind GPU 0.

Closed this session, eight arms:

| arm | reading | what it cost or bought |
|---|---|---|
| feat-103 | **GATE FAILED** | the `0.4137` it predicted against was a 100-token seed and it used 20 (caution (v)) |
| feat-104 | **T1 REPLICATES** | third pair; the metered decoder resolves at *no* budget on either judge -- the trivial horn observed |
| feat-105 | **W1 lifts, W2 flat** | TriviaQA: majority vote `+0.054`, the paper's own reward *falls* with `n`; the two rules differ in sign across tasks |
| feat-106 | **METERED WINS, FRONTIER HOLDS** | we predicted the loss before generating; it wins at `k=20` by *becoming* the risky model, certificate `480` nats |
| feat-107 | **C1 ONE GAINS, C2 NO TREND, C3 CONFIRMED** | cost us "the strongest anchor gives the largest gain"; capability buys gain *within* a family, not across |
| feat-108 | **G0 UNMATCHED** | refused the joint leakage table; residual `0.0493` in Appendix J |
| feat-109 | **INVALID** | the registered command returned the wrong novel (caution (w)) |
| feat-110 | **R1 NO LEAK** | the natural memoriser: the 70B recovers `0.2475` unaided and reproduces two of fifty passages **in full**, selection recovers `0.0000` -- Limitations drops the sentence about whose memoriser it is |

The paper is v7. It argues one claim and exhibits a mechanism on the other side of it:

> One line of the chain rule leaves a per-token budget two options. Either `K = kT` grows with the
> work and the certificate is **vacuous**, or the budget is bounded and the decoder is the anchor at
> all but `O(1)` steps --- **trivial**. Both horns are measured. The escape is not a better meter
> but a different place to spend: `q(y) <= n p_s(y)` is a pathwise certificate of exactly `log n`,
> and it does not grow with the work.

**What feat-096 settled.** Comma-7B on AlpacaEval-805 gains `+0.075 [+0.042, +0.108]` on the
registered scorer against the audited anchor's `+0.031 [-0.001, +0.062]`, and `+0.167` against
`+0.067` on judge C. **A1 CEILING CONFIRMED** across anchors; **A2 NO PROMPT-SET EFFECT**, because
judge C puts the benchmark gain *above* the in-house one and the band needed both judges below it.
This does **not** overturn feat-094: that tested the ceiling across domains *within* one anchor and
stays UNINFORMATIVE. Section 6 and Appendix J now label which axis each result speaks to, and
`results/selection_alpaca_note.md` carries the same-day partial reversal.

## State

| | |
|---|---|
| manuscript | `~/sub/satml/iclr_2027.tex`, *Vacuous or Trivial* |
| build | `exit=0`, `overfull=0`, `unresolved=0`, main text **exactly 9 of 9 pages** |
| tests | **386**, all passing |
| numeric audit | 2,179 literals, 1 expected miss (`64256`) |
| compute | `156` disclosed; **refresh after the queue drains** (`test_compute_hours.py` wants an exact match) |
| pre-registrations | **32** logs; `test_preregistration_count.py` pins the Reproducibility Statement to the count |
| tree | clean, branch `iclr-2027` |

Sections: 1 intro · 2 theory (Props 1, 2, Thm 1) · 3 onset · 4 *The dichotomy's corollaries,
tested* · 5 selection anchoring (Prop 4) · 6 experiments · 7 related · 8 limitations and conclusion.

## What is running

| job | GPU | log | state at 22:05 |
|---|---|---|---|
| feat-100 Comma-7B at `n=64` | 4 | `output/logs/sel_comma7b_64.log` | `7,400/12,800` at 23:20 |
| feat-101 judge-free axis, Comma-7B on GSM8K | 0 | `output/logs/verifiable_comma7b.log` | started 22:53, `2,624/32,000`, ~5 h |
| feat-102 selection extraction, 70B as the adversary | 1+2 | `output/logs/extraction_70b.log` | started 22:55, anchor sampling `3,288/6,400` |

A card also carries **another user's** diffusion job on GPU 2 (47 GB, 100% util) since ~21:55.
GPU 3 is the 4 GB T400 and is never used.

## Recommended next step

1. **Finish the caution (s) audit.** Scoring feat-098 found that post-EOS padding positions were
   being counted as decode steps; `dap/stats.py:strip_pad_steps` fixes it and
   `analysis/imitation_cost.py` uses it. **The other consumers of `per_step_log` have not been
   re-run**: `budget_drift.py` (`F = len(free)/len(steps)`) and `renyi_sweep.py`
   (`tot += len(per_step_log)`) both take step fractions and are the likeliest to move;
   `concentration.py`, `pathwise_price.py`, `burst_audit.py`, `regime_sweep.py` and
   `reanalyze_logs.py` need checking. Anything reading the generated *text* is unaffected.
2. **Score feat-100** when its seven `n` arms land. G3 is a nested reproducibility check and gates
   the rest: if the `n <= 8` rows disagree with the breadth arm by more than `0.03`, **G1 is not
   readable**. Under GROWS the abstract's headline number changes; under OVEROPTIMISES the
   Limitations caveat becomes a measurement.
3. **feat-101 scores itself** — `selection_verifiable.py` writes
   `results/selection_verifiable_comma7b.csv` at the end of the same run. Read V1/V2/V3 and apply
   whichever of the four V4 consequences fires; two of them weaken the constructive claim and are
   fixed in advance.
4. Refresh `analysis/compute_hours.py`, rerun `analysis/audit_numbers.py`, rebuild the artifact
   (`scripts/build_artifact.sh artifact`), and update `progress.md`.

## Still unscheduled from the scale audit

BookMIA-50 **onset** (needs memorisers re-fine-tuned, ~11 GPU-h for 3 pairs) — moderate value now
that the onset has nine pairs and two corpora. A judge-free head-to-head against the *metered*
decoder is **not possible** and the reason is on the record in
`results/onset_prediction_verifiable.md`: TinyComma is the only openly licensed anchor with the
Llama-3 tokenizer, and it scores `0.04` on GSM8K.

## Traps this session added to AGENTS.md

- **(m)** the judge is position-dominated: the same two texts win 261/500 shown second and 24 shown
  first. Never quote an absolute judged level across passes; gains survive, levels do not.
- **(n)** the page budget moves with floats, headings and table rows, not with prose.
- **(o)** `h1.py` writes the CLI `--k-values` string verbatim into filenames.
- **(p)** a gate that fails everything is not a gate; assert that something passes.
- **(q)** two 70B launch traps: a cache with no `refs/main` reports "couldn't connect" with every
  shard present, and `--parallelize` pins the risky model to one card unless `--risky-device-map
  auto --max-memory` is given.
- **(r)** every pre-registration quotes `## Scoring log` in backticks on line 3, so
  `txt.partition("## Scoring log")` splits at the wrong place. Split on the newline-prefixed
  heading.

And one that is not a trap but a habit: **render a manuscript page to PNG and look at it.** Figure 1
had been below any legibility floor for as long as it has existed, and no compile-time check saw it.
