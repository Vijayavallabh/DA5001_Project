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
  and an unparsed output is counted and excluded. **The rubric is He et al.'s own, verbatim** (their
  Table 7 with its criteria and anti-conflation rule, `analysis/he_metrics.py:FLUENCY`). They ran it
  with `gpt-4.1-mini` as the Prometheus backbone and report that open models "conflate protected
  continuations with more fluent output"; we run the open `prometheus-7b-v2.0`, on ordinary prompts
  where no protected continuation is in play, and state the substitution. It is scored on **all
  three** ordinary classes, not only on book text as He et al. do, and the class split is reported.
  An empty response is scored as the literal `(empty response)`; the mean over non-empty responses
  is reported beside it.

  *Corrected 2026-09-24, before any Prometheus call:* the first version of this note said He et al.
  do not print their rubric and used one of ours. Their appendix D.4 does print it, so it is used
  verbatim instead. No score existed when this was changed.
- **FActScore.** The `150` biography prompts (`Tell me a bio of X.`, FActScore's own entities and
  prompts, `data/factscore.jsonl`). Atomic facts at paragraph level from `Qwen2.5-14B-Instruct`,
  greedy; retrieval by BM25 over the entity's Wikipedia text (shipped in the same file) cut into
  `120`-word passages, top `5`; verification with FActScore's prompt, `... True or False?`, read off
  the `True`/`False` logits of the same model. Precision = supported / extracted; a response with no
  extracted fact is an **abstention** and is excluded from the mean, as FActScore does; the
  abstention rate and the mean number of facts are reported beside it. No length penalty. This is
  FActScore's original design (Wikipedia as the knowledge source) with open-weight models; He et al.
  instead extract and verify with `gpt-4.1-mini`, retrieve the top-5 Google snippets per claim
  (Serper), and prompt "Write a factual biography about {entity}..." on 183 entities, where our
  generations answer FActScore's "Tell me a bio of {entity}." on 150. Levels are therefore not
  comparable with theirs; only differences between our arms are read.

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

### Prometheus scored 2026-09-24 14:55 IST --- He et al.'s own fluency rubric orders the arms as the judge does

`scripts/run_he_metrics.sh prometheus` in two disjoint halves on host B GPUs 5 and 6 (`_a`, `_b`),
`analysis/he_metrics.py --report` -> `results/he_metrics.csv`. Unparsed outputs (no `[RESULT]`) are
almost all the empty responses; the reading below is the mean over **non-empty** responses, 1--5.

| arm | fluency, non-empty | n |
|---|---|---|
| `sel64` | `2.56 [2.43, 2.69]` | `447` |
| `sel1` | `1.52 [1.43, 1.62]` | `432` |
| `anchor` | `1.65 [1.55, 1.76]` | `430` |
| `met_k0.5` | `1.45 [1.37, 1.54]` | `400` |
| `met_k1` | `1.69 [1.59, 1.79]` | `400` |
| `met_k10` | `1.88 [1.78, 1.98]` | `489` |
| `risky8b` | `1.99 [1.88, 2.09]` | `496` |
| `anchor70` | `1.45 [1.37, 1.54]` | `397` |
| `met70_k0.5` | `1.42 [1.34, 1.51]` | `399` |
| `met70_k1` | `1.48 [1.39, 1.56]` | `400` |
| `met70_k20` | `1.61 [1.53, 1.70]` | `487` |
| `risky70b` | `1.62 [1.53, 1.71]` | `488` |
| `chat_k1` | `2.41 [2.28, 2.54]` | `479` |
| `chat_k10` | `3.39 [3.28, 3.50]` | `495` |
| `chat_risky` | `3.52 [3.42, 3.63]` | `496` |
| `csel64` | `3.13 [3.02, 3.24]` | `461` |
| `csel1` | `1.82 [1.71, 1.93]` | `436` |

| paired difference | fluency | n |
|---|---|---|
| `sel64-met_k10` | `+0.70 [+0.53, +0.86]` | `438` |
| `sel64-met_k0.5` | `+1.12 [+0.98, +1.28]` | `362` |
| `sel64-met_k1` | `+0.87 [+0.70, +1.03]` | `363` |
| `sel64-risky8b` | `+0.59 [+0.44, +0.75]` | `444` |
| `sel64-met70_k0.5` | `+1.15 [+1.01, +1.31]` | `362` |
| `sel64-met70_k1` | `+1.10 [+0.95, +1.26]` | `362` |
| `sel64-met70_k20` | `+0.95 [+0.80, +1.10]` | `436` |
| `sel64-risky70b` | `+0.95 [+0.80, +1.10]` | `437` |
| `sel64-chat_k10` | `-0.81 [-0.99, -0.64]` | `442` |
| `csel64-csel1` | `+1.33 [+1.17, +1.48]` | `406` |

**Reading.** On the metric He et al. use, selection at `n=64` is rated more fluent than the metered
decoder at every budget of both text-continuation configurations and than both risky models served
that way, and less fluent than the instruct model served through its chat template --- the pattern
Table 1 shows under the pairwise judge. The expected reading held. Two cautions: the backbone is the
open `prometheus-7b-v2.0` where He et al. used `gpt-4.1-mini`, and the absolute levels sit well below
their books numbers (our prompts are ordinary, and TinyComma is a weak writer), so only differences
between arms are read.
