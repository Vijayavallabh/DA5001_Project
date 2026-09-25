# Pre-registration: what selection could serve on biographies, and a scorer that asks for facts (feat-202)

**feat-202.** Committed **2026-09-25**, before any draft is decomposed or scored. Nothing above
`## Scoring log` is edited after the first FActScore call.

## Why

Review 4 (sixth round, Q8, W(e)) and Review 1 (W6, Q4): selection's FActScore precision equals its
anchor's (`0.042` against `0.048`), and "best-of-`n` is bounded by the best draw, not the average". The
appendix now says this reward "does not select for factual precision, not that a selection rule
cannot; we did not test a factuality scorer". This arm measures the best draw (the oracle) and tests a
factuality scorer, with the certificate unchanged at `log 64`.

## What runs

- **Drafts.** The committed headline pool's `150` biography prompts (`output/phase5/sel_anchor64`,
  `Tell me a bio of X.`), all `64` draws each, de-echoed (`load_candidates(..., deecho=True)`, the text
  the judge and the committed FActScore pass read): `9,600` drafts, dumped by
  `analysis/factscore_oracle.py --dump` in `analysis/he_metrics.py`'s row format, arm `d<rank>`.
- **FActScore.** `analysis/he_metrics.py --factscore` unchanged (its registered protocol,
  `results/he_metrics_note.md`: atomic facts from `Qwen2.5-14B-Instruct`, BM25 top `5` over the
  entity's Wikipedia text in `120`-word passages, FActScore's verification prompt read off the
  True/False logits), `--tag _pool64` -> `results/he_factscore_per_item_pool64.csv`. An empty draft or
  one with no extracted fact abstains and is excluded from a precision mean, as FActScore does.
- **A factuality scorer.** The committed scorer model and path (`Qwen2.5-7B-Instruct`,
  `analysis/selection_scaling.score_rewards`'s `log p(Yes) - log p(No)`), with one change, the
  question: *"Is every factual claim in this response accurate? Answer Yes or No."* in place of
  *"Is this a helpful, coherent and well-written response to the instruction?"*, over the same
  `9,600` drafts -> `results/factuality_rewards64.csv`. Selection serves the argmax, ties to the lowest
  index. Qwen2.5-7B is not the FActScore model (`14`B), so the scorer does not grade itself.
- Host B, one card, `scripts/run_feat202.sh`.

## Gates

- **G0.** `9,600` rows in the per-item file, `64` per prompt; `9,600` finite factuality rewards.
- **G1, alignment.** For all `150` prompts the draft the committed reward serves at `n=64` (argmax of
  `results/selection_rewards64.csv`, ties to the lowest index) is, character for character, the `sel64`
  text of the committed FActScore pass's dump (`output/he_metrics/arms.jsonl`).

## Readings and predictions

Means over prompts with `95%` bootstrap intervals (`10,000` resamples); paired differences on the
prompts where both picks are scored.

- **F1, the oracle.** Per prompt, the highest precision among the non-abstaining drafts in the first
  `n` (`n = 1, 8, 64`); and the same restricted to drafts with at least `5` extracted facts, since a
  draft with one supported fact has precision `1`. *Predicted: the unrestricted oracle at `n=64` is
  at least `0.25`, far above selection's `0.042`, and the restricted one is lower.*
- **F2, the factuality scorer.** Its pick's precision against the committed pick's, paired.
  *Predicted: RAISES (interval above zero).*
- **F3.** Its pick's precision against the metered decoder at `k=10` (the committed pass's `met_k10`
  rows). *Predicted: still below the meter (interval below zero).*
- **Descriptive, no band.** The mean precision of a single draw (the anchor's average); abstentions and
  mean facts per pick for each rule; the recomputed precision of the committed pick beside the
  committed pass's `sel64` row.

## What the manuscript does with each outcome, fixed now

Appendix `app:hemetrics` replaces "we did not test a factuality scorer" with F1 and F2 whatever they
read. If F2 RAISES, it says the reward chooses what selection buys and precision is within reach of a
rule that asks for it; if not, it says a scorer asking for facts did not find them, and the oracle says
whether any rule could. The main text's sentence that the head-to-head concerns fluency and helpfulness
stays, with the oracle beside it if space allows.

## Excluded in advance

- Any other factuality template, scorer model, `n`, FActScore model or retrieval setting chosen after a
  score is read; judging the factuality picks' fluency without a new registration.

## Scoring log

### Scored 2026-09-25 11:02 IST --- F1 right (the best draw reaches `0.505`), F2 WRONG (the factuality scorer ties the reward), F3 right

All jobs exited `0` on host B, GPU 7 (`scripts/run_feat202_206.sh q7`): the dump `09:16`-`09:17` IST, FActScore
over `9,475` non-empty drafts (`32,163` atomic facts) `09:17`-`10:54`, the factuality rewards `10:54`-`10:55`,
the report `10:55`. Read from `results/factscore_oracle.csv` (`.venv/bin/python analysis/factscore_oracle.py
--report`).

**Gates.** G0 PASS: `9,600` per-item rows (`64` per prompt) and `9,600` finite factuality rewards. G1 PASS: the
committed reward's pick is, character for character, the committed FActScore pass's `sel64` text on `150`
of `150` prompts; recomputed, its precision differs from that pass's by `0.0008` on average.

| reading | value | registered | verdict |
|---|---|---|---|
| F1 oracle, first `64` drafts | `0.5054 [0.4553, 0.5564]` (`150` prompts) | at least `0.25` | **right** |
| F1 oracle, drafts with `>= 5` facts | `0.2392 [0.2155, 0.2642]` | lower than unrestricted | **right** |
| F2 factuality pick minus committed pick | `+0.0102 [-0.0144, +0.0399]`, TIE (`97` prompts) | RAISES | **wrong** |
| F3 factuality pick minus the meter at `k=10` | `-0.1246 [-0.1727, -0.0725]`, BELOW (`89` prompts) | below | **right** |

**Descriptive, no band.** The oracle at `n=1` and `n=8` reads `0.0494` and `0.1764` (`0.0393` and `0.1012` with
`>= 5` facts); a single draw averages `0.0613`. The committed pick scores `0.0431` on `130` non-abstaining
biographies, abstains on `20` and claims `8.82` facts; the factuality pick scores `0.0768` on `105`, abstains
on `45` and claims `3.41`: it buys its raw precision by saying less, and paired it is not separable from the
reward's.

**Manuscript, as registered.** Appendix `app:hemetrics` replaces "we did not test a factuality scorer" with F1
and F2: the best of `64` draws is far more precise than what this reward serves (`0.5054` against `0.0431`),
and a scorer asked for accurate claims does not find it.
