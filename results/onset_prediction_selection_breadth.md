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
