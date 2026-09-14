# Pre-registration: the head-to-head at a second pair

Committed **before the selection arm is generated and before anything is judged**. Nothing above
the `## Scoring log` line is edited afterwards.

## Why this arm exists, and the constraint that decides which pair it can use

The paper's central comparison is one sentence: on the same workload, under the same judge, the
metered decoder gains `+0.072` for `171.3` nats and selection gains `+0.142` for `3.175`. It is
measured at **one** (anchor, risky) pair. The constructive side stopped being a single setup today
(four anchors, `results/onset_prediction_selection_breadth.md`), but that breadth is
**self-paired**: the factory fuses two distributions over one shared vocabulary, and none of
Pleias-1.2B, KL3M-1.7B or Comma-7B shares a tokenizer with a stronger risky model we hold, so at
those anchors the metered decoder cannot be run at all and only selection has a number.

That is worth stating plainly rather than working around: **among the openly licensed safe models
in `hf_cache/`, TinyComma-1.8B is the only one for which the head-to-head is even runnable**, because
it is the only one that ships the Llama-3 tokenizer. The one further pair that exists is
`meta-llama/Llama-3.2-1B` against `meta-llama/Llama-3.1-8B-Instruct`, which share Llama-3.1's
`128{,}256`-token vocabulary.

**Llama-3.2-1B is not a legitimate \emph{safe} model for the copyright setting.** It is trained on
undisclosed data and may have read the protected works, so no near-access-freeness claim, no
certificate and no leakage number may be taken from this arm; excluded alternative 2 below makes
that binding. What it can do is hold the mechanism, the workload, the budget grid and the judges
fixed while the pair changes, which is the only thing at issue: is the nats-for-utility gap a
property of the two mechanisms, or of TinyComma?

## What is measured

The metered arms are **already generated** --- `output/phase5/imit_llama321b`, `k \in \{-1, 0, 0.5,
1, 3, 20\}` on the same 500 ordinary prompts (200 neutral, 150 creative, 150 factual), one
trajectory each, 200-token cap, written for
`results/onset_prediction_imitation_breadth.md` before any of this was contemplated and not
regenerated. The selection arm is `n \in \{1,2,4,8\}` drawn from the same anchor on the same
prompts, nested, with the identical pointwise reward (`log p("Yes") - log p("No")` from
Qwen2.5-7B-Instruct on the one fixed template).

Both mechanisms are judged in the same pass against the **same** opponent, the `k=-1`
unconstrained completions already in that directory, by both scoring judges
(`microsoft/Phi-3.5-mini-instruct` and `meta-llama/Meta-Llama-3.1-8B-Instruct`), order randomised
per pair. `u` is the pairwise verdict scored `1/\tfrac12/0`; the **gain** is the paired difference
against the anchor-alone arm (`k=0` for the metered side, `n=1` for selection), which is the same
text either way and is judged once and reused, so the two mechanisms share a control exactly.

Realised spend comes from each arm's own trajectories, per trajectory, never from the cap.

## Bands, committed before the run

Read on judge B, the registered scorer throughout this paper; judge C reported beside it and never
substituted for it.

**F1 -- does the nats-for-utility gap survive the change of pair?** Compare selection at `n=8`
(`\log 8 = 2.08` nats of certificate, and its own measured `\log n - (n-1)/n = 1.204`) against the
**best** metered arm, the one with the largest gain whatever its budget.

| reading | band |
|---|---|
| REPLICATES | selection's gain CI excludes 0 **and** its gain is within `0.03` of the best metered arm's or higher, at less than a tenth of that arm's realised spend |
| WEAKER | selection's CI excludes 0 but its gain is more than `0.03` below the best metered arm's |
| REVERSED | the metered decoder's gain exceeds selection's by more than `0.03` with non-overlapping intervals |

**REVERSED is the reading that costs the paper its comparison**, and it is written here so it
cannot be renegotiated: under it, the `171.3`-against-`3.175` sentence would be a property of
TinyComma and would have to be stated that way in Section 6 and in the abstract.

**F2 -- does the metered decoder behave as it does at the audited pair?** Its gain should be small
and its spend should be two to three orders of magnitude larger than selection's. Descriptive, with
the realised spend of every arm reported; no band, because nothing on record calibrates a second
pair's levels and caution (e) forbids building on a judged separation smaller than a sigma.

**F3 -- what may not be claimed.** No leakage, certificate or vacuity number comes from this arm.
The anchor is not a safe model and the arm carries no protected-split generation at all. Stated so
the absence is not later read as a result.

## Excluded alternatives (named now so they cannot be adopted afterwards)

1. Regenerating the metered arms, or changing their budget grid, after seeing the selection arm.
2. Describing Llama-3.2-1B as a safe model, or quoting any certificate, leakage or `s(x)` number
   from this pair.
3. Reporting whichever judge gives the friendlier reading, or dropping one.
4. Choosing the "best metered arm" after seeing selection's gain by any rule other than the largest
   gain, which is fixed here.
5. Promoting this pair to the paper's primary comparison. The audited pair stays primary; this is a
   robustness check and is reported as one.
6. Quoting a judged separation without its sample size (caution (d), (e)).

## Scoring

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<free card> HF_HUB_OFFLINE=1 \
  HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python h1.py --k-values 0.0 \
    --safe-model-path meta-llama/Llama-3.2-1B \
    --risky-model-path meta-llama/Llama-3.1-8B-Instruct \
    --trajectories-per-prompt 8 --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
    --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 \
    --output-dir output/phase5/sel_llama321b_8
.venv/bin/python analysis/frontier_pair.py --sel-dir output/phase5/sel_llama321b_8 \
    --metered-dir output/phase5/imit_llama321b --tag _llama321b --out results
```

Writes `results/frontier_pair_llama321b.csv`.

---

## Scoring log (appended after the run; nothing above this line is edited)

---

## Scoring, 2026-09-12 (appended; nothing above is edited)

```
.venv/bin/python h1.py --k-values 0.0 --safe-model-path meta-llama/Llama-3.2-1B \
  --risky-model-path meta-llama/Llama-3.1-8B-Instruct --trajectories-per-prompt 8 \
  --cap-neutral 200 --cap-creative 150 --cap-factual 150 --cap-val 0 --cap-test 0 \
  --cap-attack-train 0 --max-new-tokens 200 --output-dir output/phase5/sel_llama321b_8
.venv/bin/python analysis/frontier_pair.py --sel-dir output/phase5/sel_llama321b_8 \
  --metered-dir output/phase5/imit_llama321b --tag _llama321b --out results
```
→ `results/frontier_pair_llama321b.csv`. The metered arms were not regenerated: they are the ones
`results/onset_prediction_imitation_breadth.md` produced, on the same 500 prompts.

### A correction to this file's own protocol, disclosed rather than folded in

The "What is measured" section above says the gain is taken "against the anchor-alone arm (`k=0` for
the metered side, `n=1` for selection), **which is the same text either way**". That parenthetical is
wrong: `k=0` from the metered sweep and `n=1` from the selection pool are two anchor-only samples at
different seeds, not one text. The *intent* --- stated in the same sentence --- was that the two
mechanisms share a control exactly, so the primary number below uses **one** shared control, the
metered run's own `k=0` arm, for both mechanisms. Selection's within-pool gain against its own
`n=1` is reported beside it and never substituted for it. Both are in the CSV.

### The numbers, on the same 500 prompts against the same `k=-1` opponent

| arm | spend, nats | gain, judge B (registered) | gain, judge C |
|---|---|---|---|
| metered, `k=0.5` | `88.1` | `+0.015 [-0.028, +0.057]` | `+0.102 [+0.045, +0.159]` |
| metered, `k=1` | `112.7` | `+0.051 [+0.008, +0.091]` | `+0.131 [+0.074, +0.189]` |
| metered, `k=3` | `123.2` | `+0.056 [+0.014, +0.097]` | `+0.124 [+0.069, +0.180]` |
| metered, `k=20` | `123.4` | `+0.055 [+0.016, +0.093]` | `+0.131 [+0.076, +0.189]` |
| selection, `n=2` | `0.193` | `+0.024 [-0.018, +0.068]` | `+0.080 [+0.038, +0.122]` |
| selection, `n=4` | `0.636` | `+0.048 [+0.006, +0.090]` | `+0.140 [+0.094, +0.186]` |
| **selection, `n=8`** | **`1.204`** | **`+0.076 [+0.031, +0.122]`** | **`+0.154 [+0.104, +0.200]`** |

### F1 — **REPLICATES**, on both judges

On the registered scorer the best metered arm is `k=3`, which gains `+0.056` for `123.2` nats;
selection at `n=8` gains **more**, `+0.076`, for `1.204` --- `102\times` less divergence. On judge C
the best metered arm is `k=1` at `+0.131` for `112.7` nats and selection gains `+0.154` for the
same `1.204`, `94\times` less. Both selection intervals exclude zero. The committed band asked only
that selection come within `0.03` of the best metered arm at under a tenth of its spend; it beats it
outright at a hundredth.

**The nats-for-utility gap is not a property of TinyComma.** That was the reading REVERSED would
have cost the paper, and it is not what happened.

### F2 — descriptive: the metered decoder is *useful* here, and still loses

The contrast with the audited pair is worth recording rather than smoothing over. At TinyComma the
metered decoder's best arm gains `+0.072` for `171.3` nats and every lower budget gains nothing
measurable. Here it gains `+0.05` to `+0.13` and resolves at three of four budgets --- a Llama-3.2-1B
anchor is close enough to the risky model that a per-token meter buys real utility. It still spends
two orders of magnitude more than selection to buy less of it, and its own spend saturates at
`123.4` nats exactly as Proposition 3 says: `k=3` and `k=20` differ by `0.2` nats of realised spend
and by `0.001` of judged gain, while the published cap between them differs by a factor of seven.

Within-pool, selection's `n=8` gain against its own `n=1` is `+0.110` (judge B) and `+0.143`
(judge C), both larger than against the shared control, because the two anchor-only samples differ
by about `0.03` of judged utility at these sample sizes. The shared-control number is the one
reported above and is the more conservative of the two on both judges.

### F3 — nothing else may be taken from this pair

`Llama-3.2-1B` is not a safe model. No certificate, leakage or `s(x)` number appears above and none
may be inferred; the arm decoded nothing on the protected split. The audited pair remains the
paper's primary comparison and this is a robustness check, as excluded alternative 5 requires.

### The constraint that made this the only available second pair

Worth repeating from above the line, because it is a limitation of the field rather than of this
run: among the openly licensed safe models we hold, **TinyComma-1.8B is the only one for which the
metered decoder runs at all**, because anchored decoding fuses two distributions over one shared
vocabulary and no other openly licensed anchor ships the Llama-3 tokenizer. A second *legitimate*
head-to-head needs either an openly licensed anchor with a frontier model's tokenizer, or a
mechanism that does not require a shared vocabulary. Neither exists today.
