# Pre-registration: does the constructive claim hold at more than one anchor?

Committed **before any arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

The paper's audit half is broad: nine (anchor, risky) pairs with nine distinct anchors and six
tokenizers for the onset, twelve pairs in seven families for the matched-utility law, more than
100,000 budgeted trajectories, two protected corpora. The **constructive half is not**. Every
selection-anchoring number on record --- the `+0.081` cross-judge gain, the `0.0000` recall, the
`AUC 0.526` diagnosis, the `332`-against-`2` composition --- comes from **one anchor**,
TinyComma-1.8B, on **one** prompt set of 500. The v7 reframe made that half load-bearing, so a
single setup is now the narrowest evidence in the paper, and a reviewer is right to call it
anecdotal.

This arm runs the same mechanism at three further anchors spanning a `4\times` range in parameters
and three model families, all openly licensed and already in `hf_cache/`, against the **same**
unconstrained Llama-3.1-8B-Instruct baseline on the **same** 500 prompts under the **same** judge, so
the anchors are the only thing that changes.

| anchor | params | family | why |
|---|---|---|---|
| `PleIAs/Pleias-1.2b-Preview` | 1.2B | Pleias | smaller than the audited anchor |
| `alea-institute/kl3m-003-1.7b` | 1.7B | KL3M | matched size, different corpus and tokenizer |
| `common-pile/comma-v0.1-2t` | 7B | Comma | the strongest openly licensed anchor we hold |

`n \in \{1,2,4,8\}` (nested, so no candidate is scored twice), scored by judge B
(`microsoft/Phi-3.5-mini-instruct`) which does no selecting, with judge A (`Qwen2.5-7B-Instruct`)
selecting --- the identical protocol that produced `+0.081` on TinyComma.

## Entry gate

An anchor enters only if its `n = 1` arm produces non-degenerate text: mean generation length
`> 20` tokens and fewer than 5% empty completions. An anchor that cannot write is not evidence
about selection, and admitting one would let a broken arm masquerade as a null.

## Bands, committed before the run

**B1 -- does the gain generalise?** Per anchor, the paired `n=8` minus `n=1` gain with a 95%
bootstrap CI over prompts. TinyComma's is `+0.081 [0.034, 0.130]`.

| reading | band |
|---|---|
| GENERALISES | at least 2 of the 3 anchors have a gain whose CI excludes 0 |
| PARTIAL | exactly 1 does |
| SINGLE-SETUP | none does |

**SINGLE-SETUP is the reading that costs the paper its constructive claim**, and under it the
manuscript must narrow: selection anchoring would be reported as measured on the audited pair only,
Section 6's generality sentences removed, and the abstract's "reaches the judged utility a metered
decoder needs 171.3 nats for" qualified to that pair. That is written here so it cannot be
renegotiated afterwards.

**B2 -- is the anchor's capability the ceiling, as Limitations claims?** Report Spearman of the
anchor-alone utility `u(n{=}1)` against the gain across the four anchors (three new plus TinyComma),
with its exact permutation `p`. Four points cannot establish a trend; this is descriptive and is
reported as such, never as support for the ceiling argument.

**B3 -- leakage, mandatory.** Near-verbatim recall on the protected passages at every anchor and
every `n`, with the `k=-1` and `k=0` baselines on the same prompts and seeds. TinyComma's is
`0.0000` at every `n \le 64`. **Any non-zero recall at any anchor is reported in the main text**,
whatever it does to the utility story: the leakage claim is the stronger of the two and must not be
protected.

## Excluded alternatives (named now so they cannot be adopted afterwards)

1. Dropping an anchor after seeing its gain. The entry gate above is the only admissible exclusion
   and is stated before the run.
2. Reporting the best anchor as "the" result, or the mean across anchors without the per-anchor
   spread.
3. Changing the judge, the selector, the prompt set, the baseline or `n` between anchors.
4. Substituting an instruction-tuned anchor to raise the level. Every anchor here is a base model
   trained on openly licensed text, which is what a safe model in this setting is.
5. Treating B2 as evidence for the support-ceiling argument on four points.
6. Quoting a judged gain without its sample size, or building on a separation smaller than a sigma
   (cautions (d) and (e)).

## Scoring

```
# per anchor: 8 anchor-only samples on the same 500 ordinary prompts
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<free card> HF_HUB_OFFLINE=1 \
  HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python h1.py --k-values 0.0 \
  --safe-model-path <anchor> --risky-model-path meta-llama/Llama-3.1-8B-Instruct \
  --trajectories-per-prompt 8 --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 \
  --output-dir output/phase5/sel_<tag>8
.venv/bin/python analysis/selection_breadth.py --out results
```

Writes `results/selection_breadth.csv` and `results/selection_breadth_per_prompt.csv`.

---

## Scoring log (appended after the run; nothing above this line is edited)

---

## Scoring, 2026-09-12 (appended; nothing above is edited)

```
# per anchor, self-paired so the factory's shared-vocabulary requirement holds (at k=0 only the
# safe model generates, so the risky slot is inert):
.venv/bin/python h1.py --k-values 0.0 --safe-model-path <anchor> --risky-model-path <anchor> \
  --trajectories-per-prompt 8 --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 \
  --output-dir output/phase5/sel_<tag>_8
.venv/bin/python analysis/selection_scaling.py --gen-dir output/phase5/sel_<tag>_8 \
  --tag _<tag> --out results
.venv/bin/python analysis/selection_breadth.py --out results
```

Same 500 ordinary prompts, same unconstrained Llama-3.1-8B-Instruct baseline, same pointwise
reward, same two judges, `n=8`. The anchor is the only thing that changes.

| anchor | params | family | `u(n{=}1)` | gain, judge B (registered) | gain, judge C | gate |
|---|---|---|---|---|---|---|
| TinyComma-1.8B (audited) | 1.8B | Comma | `0.435` | `+0.054 [+0.013, +0.095]` | `+0.073 [+0.027, +0.120]` | **FAIL** |
| Pleias-1.2B | 1.2B | Pleias | `0.411` | `+0.029 [-0.012, +0.068]` | `+0.071 [+0.027, +0.115]` | PASS |
| KL3M-1.7B | 1.7B | KL3M | `0.310` | `+0.039 [+0.005, +0.076]` | `+0.051 [+0.013, +0.090]` | PASS |
| Comma-7B | 7B | Comma | `0.450` | `+0.111 [+0.072, +0.148]` | `+0.155 [+0.106, +0.200]` | PASS |

### B1 — **GENERALISES**

Two of the three new anchors --- KL3M-1.7B and Comma-7B --- have a gain whose CI excludes zero on
the registered scorer, which is the committed band for GENERALISES. On the secondary judge all
three do. Pleias-1.2B is positive on both judges and resolves on one.

The constructive claim is no longer a single setup: **four anchors, three model families, `1.2`B to
`7`B, all openly licensed base models, one prompt set, one judge, one baseline.** The band was
written so that SINGLE-SETUP would cost the paper its constructive claim; it does not read
SINGLE-SETUP.

The largest gain in the paper is now **Comma-7B's `+0.111` (judge B) and `+0.155` (judge C)**, at
the same `\log 8 = 2.0794` nats --- the strongest openly licensed anchor we hold, and half again
the audited anchor's.

### The entry gate fails the *audited* anchor, and that is reported rather than exempted

The gate is "mean generation length `> 20` tokens and fewer than 5% empty completions". Measured on
each arm's own `n=1` generations: every anchor writes about 200 tokens, and the empty fractions are
`0.0%` (Pleias), `0.2%` (KL3M), `3.0%` (Comma-7B) and **`6.8%` (TinyComma)**. The audited anchor ---
the one every number in the paper before today came from --- is above its own threshold.

Two things follow, and both are done rather than argued. First, the failure is recorded in
`results/selection_breadth.csv` and in AGENTS caution (p), not exempted; the first implementation
of this gate read a column its producer never writes and marked *every* anchor FAIL, so it had
never actually run. Second, the obvious worry is answered directly: best-of-`n` never selects an
empty candidate, so part of the gain could be nothing but a degeneracy filter. Recomputed on the
prompts whose `n=1` completion is non-empty, it is not:

| anchor | gain, judge B | on non-empty only |
|---|---|---|
| TinyComma-1.8B | `+0.0540` | `+0.0536 [+0.0075, +0.0966]` (466) |
| Pleias-1.2B | `+0.0290` | `+0.0290 [-0.0120, +0.0690]` (500) |
| KL3M-1.7B | `+0.0390` | `+0.0391 [+0.0040, +0.0752]` (499) |
| Comma-7B | `+0.1110` | `+0.1124 [+0.0722, +0.1505]` (485) |

Every gain moves by less than `0.003`, and every CI that excluded zero still does.

### B2 — Spearman `+1.000` over four anchors, **descriptive and not used**

The anchor-alone level and the gain are perfectly rank-correlated across the four anchors, in the
direction the support-ceiling argument predicts, and against the shared-noise bias (which subtracts
`u(n{=}1)` inside the gain and therefore pushes this negative). Exact permutation `p = 0.083`
two-sided on four points.

**This is registered as descriptive and excluded alternative 5 forbids using it as ceiling
evidence, so it is not used as any.** Four points cannot establish a trend, and the domain split
run today (`results/onset_prediction_domain_breadth.md`) came back uninformative on the same
question along a different axis. What the two together justify is running the deciding arm rather
than arguing about it: Comma-7B on AlpacaEval-805, pre-registered in
`results/onset_prediction_alpaca_comma7b.md` **before** it was launched, with the manuscript
consequence of each reading fixed there.

### B3 — leakage: running, appended below when it lands

`analysis/selection_extraction.py` at each new anchor, `n \in \{1,8,64\}`, 100 protected passages,
adversarial selector (the memorising model's own likelihood), with the mandatory baselines. One
defect had to be fixed first and is worth recording: the script fed the **safe** model's token ids
to *both* models, which was harmless only because the audited anchor ships the Llama-3 tokenizer
the memoriser also uses, and would have been silently wrong at every anchor added today. Each model
now gets its own tokenizer, and the 20-token seed is built with the memoriser's tokenizer at every
anchor so the seed is byte-identical across them --- verified to reproduce the audited pair's
passages exactly, all 100 seeds and all 100 targets.
