# Note: He et al.'s own utility metrics on this paper's served texts (feat-191; descriptive)

Committed **2026-09-24, before any Prometheus or FActScore call.** Nothing above `## Scoring log` is
edited afterwards. Descriptive: no bands, because nothing here was predicted by the paper, and every
reading is the interval as the script writes it.

## Why

Referees: He et al. (arXiv 2602.07120) measure utility with **Prometheus-v2 fluency** (a five-point
rubric, on book continuations) and **FActScore claim precision** (on biographies), and this paper
reports neither --- only a pairwise judge. `analysis/he_metrics.py` scores the same de-echoed texts the
judge reads, arm for arm, with both.

## Arms (`analysis/levels_pass.py` specs; one text per prompt, the headline's `500` prompts)

Released-8B block: `sel64`, `sel1` (selection at `n=64`, `n=1`, TinyComma pool, cached picks),
`anchor` (`sweep_plain` `k=0`), the meter `met_k0.5`, `met_k1`, `met_k10` (`conc_all`), `risky8b`
(`sweep_plain` `k=-1`, the committed opponent). 70B block: `anchor70`, `met70_k0.5`, `met70_k1`,
`met70_k20`, `risky70b` (`imit_llama70b`). Chat block: `chat_k1`, `chat_k10`, `chat_risky`. Comma-7B:
`csel64`, `csel1`. The AnchoredByte arms of feat-187 are added under their own `--tag` when generated.

## Protocol, fixed now

- **Prometheus.** `prometheus-eval/prometheus-7b-v2.0`, the library's **no-reference** absolute
  template and system prompt, verbatim; the instruction shown is the corpus prompt (header stripped,
  as the judge sees it); greedy decoding, `384` new tokens; the score is the integer after `[RESULT]`,
  and an unparsed output is counted and excluded. **The rubric is ours**, because He et al. do not
  print theirs: fluency and coherence, 1 (incoherent or empty) to 5 (fully fluent), in
  `analysis/he_metrics.py:FLUENCY`. It is scored on **all three** ordinary classes, not only on book
  text as He et al. do, and the class split is reported.
- **FActScore.** The `150` biography prompts (`Tell me a bio of X.`, FActScore's own entities and
  prompts, `data/factscore.jsonl`). Atomic facts at paragraph level from `Qwen2.5-14B-Instruct`,
  greedy; retrieval by BM25 over the entity's Wikipedia text (shipped in the same file) cut into
  `120`-word passages, top `5`; verification with FActScore's prompt, `... True or False?`, read off
  the `True`/`False` logits of the same model. Precision = supported / extracted; a response with no
  extracted fact is an **abstention** and is excluded from the mean, as FActScore does; the
  abstention rate and the mean number of facts are reported beside it. No length penalty. This is
  FActScore's protocol with open-weight models, not its original InstructGPT/ChatGPT pipeline, and
  it is stated so.

## Readings

Per arm: mean and 95% bootstrap interval over prompts. Paired differences, the ones Table 1 reads:
`sel64-met_k10`, `sel64-met_k0.5`, `sel64-met_k1`, `sel64-risky8b`, `sel64-anchor`,
`sel64-met70_k0.5`, `sel64-met70_k1`, `sel64-met70_k20`, `sel64-risky70b`, `csel64-csel1`, and for the
AnchoredByte arms `csel64-ab_k*`.

**Expected, not banded:** selection's gain should show on Prometheus fluency, whose rubric resembles
what the reward model rewards; on FActScore precision a best-of-`n` over a `1.8`B anchor has no source
of facts the anchor lacks, so selection should gain little over `sel1` and lose to any arm that serves
mostly the risky model's tokens. Either outcome is reported as measured.

## Excluded in advance

- Changing the rubric, template, model, retrieval or `k` after reading any score.
- Reading a Prometheus level as a statement about book text alone.

## Scoring log
