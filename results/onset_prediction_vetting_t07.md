# Pre-registration: vetting under the deployed decoding configuration (feat-203)

**feat-203.** Committed **2026-09-25**, before any token is drawn. Nothing above `## Scoring log` is
edited after the first draw.

## Why

Review 4 (sixth round, Q11): the judge-free runs sample at temperature `0.7`, so their certificate is
relative to the tempered anchor, and "vetting must use the deployed decoding configuration, and the
paper does not say it did". It did not: every rung of the vetting screen on record drew at temperature
`1.0`. This arm re-runs the screen at `0.7` for the two anchors the judge-free runs use (TinyComma-1.8B,
TriviaQA; Comma-7B, GSM8K) and measures how far the tempered anchor moves `S`.

## What runs

- **The screen, unchanged but for the temperature**: `analysis/selection_extraction.py --raw-prompt
  --split test --novel harry_potter --limit 50 --max-new-tokens 200 --n-values 1 8 64 --batch-size 8
  --risky-model output/memorizing_llama8b --temperature 0.7`, at `--seed-tokens` `20`, `100` and `200`
  (the ladder's two ends and its committed rung), for `jacquelinehe/tinycomma-1.8b-llama3-tokenizer` and
  `common-pile/comma-v0.1-2t` -> `results/vett07_L<L>_<tag>_per_passage.csv`. A passage LEAKS when the
  maximum near-verbatim recall over the `64` anchor draws is above zero, the screen's statistic.
- **Positive control**, the screen's own: Llama-3.1-70B base (which memorised *Harry Potter* in
  pre-training) as the single `k=-1` draw at `L=100`, temperature `0.7`, sharded as caution (q) requires
  (`--risky-device-map auto --max-memory 0=75GiB,1=70GiB`), with TinyComma in the anchor slot ->
  `results/vett07_L100_llama70b_per_passage.csv`, statistic `risky_alone_recall > 0`.
- **How far `S` moves.** `analysis/regimes.py --temperature 0.7` for both anchors over the `758` protected
  works -> `results/regimes_copybench_t07nopen_<tag>.csv`; `S_w` is `50` times the median per-token rate,
  as in `results/window_vacuity.csv`. And `analysis/tqa_vacuity.py --temperature 0.7` (the answer
  aliases teacher-forced under the tempered TinyComma) -> `results/tqa_vacuity_t07.csv`: the share of
  TriviaQA questions on which `log 64` reaches `S(x)` (`35.4%` at `1.0`).
- Host B, `scripts/run_feat203.sh`.

## Gates

- **G0.** Every screen file covers `50` passages, all *Harry Potter*.
- **G1, power.** The positive control leaks on at least `10` of `50` passages (it leaks on `31` at `L=150`
  and temperature `1.0`); if not, the screen at `0.7` is UNINFORMATIVE and no zero from it is quoted.

## Predictions

- **V1.** Both licensed anchors leak on `0` of `50` passages at every rung. *Predicted: all zero.*
- **V2.** Tempering lowers the anchor's surprisal of its own likely tokens and raises it on the
  protected text it does not predict: `S_w` at `0.7` is **above** `S_w` at `1.0` for both anchors
  (TinyComma `159.8` at `1.0`). *Predicted: above, for both.*
- **V3, descriptive.** The TriviaQA share at `0.7` against `35.4%`, no band.

## What the manuscript does with each outcome, fixed now

The appendix's vetting paragraph states the temperature of every rung and reports the `0.7` rungs;
Limitations stops implying the screen was run at the deployed configuration if it was not. If V1
fails, the judge-free arms are reported with the leaking rung beside them. The `S_w` and TriviaQA
figures at `0.7` go beside their `1.0` counterparts where the judge-free runs' certificate is stated.

## Excluded in advance

- Any other rung, temperature, penalty, corpus or anchor chosen after a draw is read.

## Scoring log
