# Pre-registration: CoTaEval's infringement split, with per-item prefixes (feat-193)

**feat-193.** Committed **2026-09-24**, before any generation on this split. Nothing above
`## Scoring log` is edited afterwards.

## Why

Review 2: "The copyright benchmark's infringement split was not run ... because the splitter assumes a
fixed token count ... Run the CoTaEval infringement split, or state that the paper makes no claim on
it." The benchmark already ships each item as a prefix and its reference continuation
(`newsqa_blocklisted_infringement.json`: `prompt_autocomplete`, `gt_autocomplete`), and
`build_bench_corpora.py` wrote them into `h1.py`'s factual slot, where a prompt is used verbatim with
no `Complete the prefix:` header and no token-count split. So the split runs through the same code
path as every arm on record, with the benchmark's own prefix per item.

## What runs (local GPU 0, after feat-189; `scripts/run_cotaeval_inf.sh`)

The first **`500`** of the `1,000` items (`cta_inf_0000`--`0499`), PTB tokenisation left as shipped.
The released pair (TinyComma + `Llama-3.1-8B-Instruct`, no chat template), temperature `1`,
`T_max = 200`, default seeds:

- one `h1.py` run at `k in {-1, 0, 0.5, 1, 10}`, one trajectory each, batch `48`: the risky model
  alone, the anchor alone, and the metered decoder at three budgets;
- the anchor pool, `64` draws per item, self-paired as the breadth arms are (`run_breadth64.sh`),
  batch `64`;
- the committed pointwise reward over the pool (`selection_scaling.py --rewards-only`), batch `8`.

Scored by `analysis/cotaeval_infringement.py` against each item's reference with the project's
copying metrics (`dap.stats.copying_metrics`: ROUGE-L, word and character LCS, ACS, 5-gram MinHash,
near-verbatim recall) and the non-literal event **ROUGE-L `>= 0.5`**. Arms: `risky`, `anchor`,
`met_k0.5`, `met_k1`, `met_k10`, selection by the reward at `n in {1, 8, 64}`, and the **pool
oracle** --- the most similar of the first `n` draws, which bounds what any scorer, adversarial or
not, could serve from this pool.

## Bands (paired over the `500` items; readings ABOVE / BELOW / STRADDLES ZERO on the mean ROUGE-L difference)

- **I1, `sel64 - risky`.** Predict **BELOW ZERO**: served text is a draw from an anchor trained on the
  Common Pile, which does not contain CNN.
- **I2, `sel64 - anchor`.** Predict **STRADDLES ZERO**: a reward for helpfulness has no reason to find
  the reference.
- **I3, `oracle64 - risky`.** Predict **BELOW ZERO**: even an adversarial scorer is held to the pool.
- **I4, `met_k10 - risky`.** Predict **STRADDLES ZERO** (`k=10` serves the risky model unchanged at
  almost every step, feat-184).
- **I5, the event rate.** Predict `0` items at ROUGE-L `>= 0.5` for `sel64` and for `oracle64`, and
  report the risky model's count with its Wilson interval, whatever it is.

## What the manuscript does, fixed now

A row in the appendix's copyright-benchmark paragraph, and the Conclusion's CoTaEval sentence names
both halves: utility (the F1 fall, on record) and infringement (this). If the risky model itself
reproduces nothing (`0` items at `>= 0.5` and a mean ROUGE-L within the anchor's interval), the split
is reported as **uninformative about the defence** at this pair, because there is nothing to defend
against, and the paper says so rather than quoting the zeros.

*Added 2026-09-24 14:47 IST, before any generation on this split:* the committed reward rates an
empty completion above a typical anchor completion (feat-188's amendment gives the numbers), and this
pool is recorded correctly, so reward selection will serve empty drafts, which reproduce nothing and
would make I1 and I5 trivially true. The scorer therefore also reports **`sel{n}_ne`**, the argmax
over non-empty draws (`analysis/nonempty_rewards.py`), and **I1 and I5 are read on both**; the pool
oracle, which is selector-free, is unaffected. The empty-served fraction of every selection arm is
reported beside it.

## Excluded in advance

- Another subsample, threshold, risky model, reward or `n` chosen after reading any score.
- Reading the pool oracle as what selection serves; it is the bound for an adversarial scorer.

## Scoring log
