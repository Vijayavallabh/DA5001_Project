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

### Scored 2026-09-24 17:07 IST --- I1, I1b, I2, I4 and I5 as predicted; I3 WRONG; the uninformative rule does NOT fire

Generation `scripts/run_cotaeval_inf.sh` and `scripts/run_cotaeval_inf_split.sh` (local GPUs), the
reward over the pool in four prompt shards (`scripts/run_cotaeval_reward_shards.sh`; batch `8` divides
`64`, so every batch holds one prompt's candidates, and the merged cache equals the union of the four
shards row for row, `32,000` of `32,000`), then `analysis/cotaeval_infringement.py --out results` ->
`results/cotaeval_infringement.csv`. The scorer's metric loop was parallelised before this run (a pure
function of text and reference, same bootstrap stream); the first, single-process scorer timed out
and wrote nothing (`cta_score.fail`).

| band | contrast (mean ROUGE-L, paired over `500`) | predicted | measured | reading |
|---|---|---|---|---|
| I1 | `sel64 - risky` | BELOW ZERO | `-0.0150 [-0.0195, -0.0107]` | as predicted |
| I1b | `sel64_ne - risky` | BELOW ZERO | `-0.0148 [-0.0191, -0.0104]` | as predicted |
| I2 | `sel64 - anchor` | STRADDLES ZERO | `-0.0012 [-0.0052, +0.0027]` | as predicted |
| (I2b) | `sel64_ne - anchor` | not banded | `-0.0010 [-0.0048, +0.0031]` | straddles |
| **I3** | `oracle64 - risky` | BELOW ZERO | `+0.0370 [+0.0336, +0.0404]` | **WRONG: ABOVE ZERO** |
| I4 | `met_k10 - risky` | STRADDLES ZERO | `+0.0004 [-0.0002, +0.0010]` | as predicted |
| I5 | items at ROUGE-L `>= 0.5` | `0` for `sel64`, `sel64_ne`, `oracle64` | `0/500` at **every** arm | as predicted |

Reported, not banded: `met_k0.5 - risky` `-0.0145 [-0.0185, -0.0105]`; `sel64 - met_k10`
`-0.0153 [-0.0199, -0.0109]`. Levels: risky `0.1589 [0.1554, 0.1624]`, anchor `0.1451 [0.1421,
0.1481]`, `sel64` `0.1439`, `oracle8` `0.1783`, `oracle64` `0.1960 [0.1939, 0.1981]`. The risky
model's event count is `0/500`, Wilson `[0, 0.0076]`, and so is every other arm's. Reward selection
served an empty draft on `1` item of `500` at `n=64` (`empty_served 0.002`) and on none at `n<=8`;
`sel64_ne` reads the same to three decimals.

**The uninformative rule does not fire, and the paper does not use its wording.** It required BOTH
`0` risky items at `>= 0.5` AND a risky mean inside the anchor's interval; the first holds and the
second does not (`0.1589` against `[0.1421, 0.1481]`). So the split is not "uninformative about the
defence": its **event** half separates nothing at this pair (no arm, the unconstrained `8`B model
included, reproduces a single item at the benchmark's threshold), while its **mean** half does
separate --- the risky model sits closer to the reference than the anchor, and every defence that
serves anchor text (selection at `n=64`, with or without empties, the meter at `k=0.5`) sits with the
anchor.

**I3, why it was wrong.** The registration argued that an adversarial scorer "is held to the pool" and
would therefore stay below the risky model. Proposition 1 holds it to the pool relative to the
**anchor** --- `q(y) <= n p_s(y)`, amplification by at most `n` --- and says nothing about the risky
model. The oracle is a maximum over `64` draws of a per-item similarity whose floor is shared common
words, and such a maximum exceeds one draw of a model only `0.014` closer on average. It is also the
amplification the certificate prices, made visible: `oracle1 = sel1 = 0.1428`, `oracle8 = 0.1783`,
`oracle64 = 0.1960`. On the near-verbatim metrics the oracle is not distinguishable from the risky
model (`nv_recall 0.0016 [0.0002, 0.0036]` against `0.0010 [0, 0.0025]`) while its 5-gram MinHash is
higher (`0.0040 [0.0034, 0.0047]` against `0.0016 [0.0012, 0.0020]`); and even the oracle reaches the
event on no item. What the paper may say: a best-of-`64` adversary serving from this anchor's pool
gets closer to CNN's text on average than the unconstrained `8`B model, and never to the benchmark's
threshold on any of `500` items.

**What the manuscript now says, as fixed above.** The appendix's `Not run` sentence is replaced by the
result with both halves and the failed prediction; the Limitations clause that cites CoTaEval names
both halves (F1 falls; no arm, the risky model included, reaches the infringement event).
