# Pre-registration: the deciding arm — Comma-7B on AlpacaEval-805

Committed **before the arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## What is on record, stated first

| arm | judge B (registered scorer) | judge C |
|---|---|---|
| TinyComma-1.8B, in-house 500 | `+0.054 [+0.013, +0.095]` | `+0.073 [+0.027, +0.120]` |
| TinyComma-1.8B, AlpacaEval-805 | `+0.031 [-0.001, +0.062]` | `+0.067 [+0.034, +0.101]` |
| TinyComma-1.8B, MT-Bench-80 | `+0.006 [-0.088, +0.094]` | `-0.038 [-0.131, +0.050]` |
| Comma-7B, in-house 500 | `+0.111 [+0.072, +0.148]` | `+0.155 [+0.106, +0.200]` |

Two questions are open and this one arm separates them, which is why it is worth the compute.

1. **Is the weaker standard-benchmark gain the anchor's support ceiling?** The obvious reading of
   the AlpacaEval drop is that a `1.8`B base anchor cannot follow instructions, so a selector has
   less to reorder. The pre-registered domain split
   (`results/onset_prediction_domain_breadth.md`) tested that *within* TinyComma across seven
   domain cells and came back **uninformative** --- its `-0.79`/`-0.70` is inseparable from a
   no-effect null already giving `-0.36 +/- 0.34`. So the ceiling is neither shown nor ruled out.
2. **Or is our in-house prompt set simply flattering?** Every headline number was measured on 500
   prompts we wrote. If the gain drops on a standard benchmark at *every* anchor, that is a
   property of the prompt set, and the paper must lead with the benchmark number.

`Comma-7B` is the anchor that separates them: it is the strongest openly licensed base model we
hold, and on the in-house set it gives the **largest** gain in the paper. If a stronger anchor
recovers the gain on AlpacaEval, reading 1 is supported. If it drops there exactly as TinyComma
did, reading 2 is, and the strength of the anchor is not what is limiting.

Note that B2 of `results/onset_prediction_selection_breadth.md` reads Spearman `+1.000` over four
anchors between the anchor-alone level and the gain, in the direction reading 1 predicts and
against its own shared-noise bias. **That is four points and is registered as descriptive only**
(excluded alternative 5 there forbids using it as ceiling evidence). It is a reason to run this
arm, not a substitute for it.

## What is run

`data/bench/alpaca/`, the full AlpacaEval instruction set of 805 prompts through the **factual**
slot, exactly as the TinyComma AlpacaEval arm ran it: `n \in \{1,2,4,8\}` nested, the identical
pointwise reward (`log p("Yes") - log p("No")` from Qwen2.5-7B-Instruct on the one fixed template),
both scoring judges, `400`-token cap, judged against the **same** unconstrained
Llama-3.1-8B-Instruct completions already generated for that arm
(`output/phase5/alpaca_risky`), so the baseline is byte-identical across anchors. The only thing
that changes is the anchor.

The entry gate of `results/onset_prediction_selection_breadth.md` applies unchanged: mean
generation length `> 20` tokens and fewer than 5% empty completions on the `n=1` arm. Comma-7B is
at 3.0% on the in-house set and so is expected to pass, but the gate is measured on this arm's own
generations and reported either way.

## Bands, committed before the run

**A1 -- does a stronger anchor recover the standard-benchmark gain?** Comma-7B on AlpacaEval at
`n=8`, against TinyComma's `+0.031` there. Read on judge B, the registered scorer; judge C
reported beside it and never substituted for it.

| reading | band |
|---|---|
| CEILING CONFIRMED | the gain CI excludes 0 **and** the point estimate exceeds `+0.031` by more than `0.03` |
| RECOVERS, CEILING NOT SHOWN | the CI excludes 0 but the gain is within `0.03` of `+0.031` |
| NO RECOVERY | the CI includes 0 |

**A2 -- is the in-house prompt set flattering?** The same anchor's AlpacaEval gain against its own
in-house gain (`+0.111` judge B, `+0.155` judge C). This is the within-anchor comparison and is the
cleaner of the two, because nothing but the prompts changes.

| reading | band |
|---|---|
| PROMPT-SET EFFECT | the AlpacaEval gain is below the in-house gain by more than `0.03` in **both** judges |
| NO PROMPT-SET EFFECT | within `0.03` in at least one judge |

A1 and A2 can both fire, and that combination is meaningful rather than contradictory: it would say
a stronger anchor helps on instructions *and* that our own prompts still flatter the mechanism. The
manuscript consequence is fixed now: **under PROMPT-SET EFFECT the paper leads with the AlpacaEval
number and reports the in-house one beside it**, in Section 6 and in the abstract, whatever A1 says.

**A3 -- leakage.** AlpacaEval carries no protected passages, so no leakage number comes from this
arm and none may be inferred from it. The `0.0000` recall claim stands on the protected split at
every `n <= 64` and is unchanged by anything here. Stated so the absence is not read as a result.

## Excluded alternatives (named now so they cannot be adopted afterwards)

1. Changing `n`, the judge, the selector, the reward template, the baseline completions or the
   token cap between this arm and the TinyComma AlpacaEval arm.
2. Reporting whichever judge resolves and omitting the other.
3. Dropping prompts, or restricting to a subset of AlpacaEval's five sources, after seeing the
   numbers. The entry gate is the only admissible exclusion.
4. Treating A1 as settling the domain-level ceiling question of
   `results/onset_prediction_domain_breadth.md`. That is a different axis --- domains within one
   anchor, against anchors on one prompt set --- and D1 stays uninformative whatever A1 reads.
5. Quoting the in-house `+0.111` as the paper's headline if A2 reads PROMPT-SET EFFECT.
6. Re-describing MT-Bench's null as anything but underpowered at 80 prompts.

## Scoring

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<free card> HF_HUB_OFFLINE=1 \
  HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python h1.py --k-values 0.0 \
  --safe-model-path common-pile/comma-v0.1-2t \
  --risky-model-path meta-llama/Llama-3.1-8B-Instruct --data-dir data/bench/alpaca \
  --trajectories-per-prompt 8 --cap-factual 805 --cap-neutral 0 --cap-creative 0 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 400 \
  --output-dir output/phase5/alpaca_comma7b_8
.venv/bin/python analysis/selection_scaling.py --gen-dir output/phase5/alpaca_comma7b_8 \
  --baseline-dir output/phase5/alpaca_risky --tag _alpaca_comma7b --out results
```

---

## Scoring log (appended after the run; nothing above this line is edited)
