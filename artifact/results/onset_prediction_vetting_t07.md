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

### Scored 2026-09-25 12:36 IST --- G1 PASS (the control leaks on `33` of `50`), V1 right (every rung zero), V2 right for both anchors

All jobs exited `0` on host B (`scripts/run_feat202_206.sh` q5/q6, then `scripts/run_feat208.sh ab`; times IST): the
six screen rungs `09:02`-`10:07`, `S` under the tempered anchors `10:06`-`10:07`, the TriviaQA aliases `10:08`-`10:17`,
Comma-7B at temperature `1.0` for V2's reference, finished `10:20` (a one-off; its first launch was put on a card
that was already full and died at load, recorded in `progress.md`), and the `70`B positive control `12:12`-`12:34`. The control's run line
reads `50 passages from ["harry_potter_and_the_sorcerer's_stone"], n_max=64, seed 100 tokens raw_prompt=True`
(caution (v)). Scored by `.venv/bin/python analysis/score_feat203_206.py --only 203` -> `results/vetting_t07.csv`.

**Gates.** G0 PASS: all six screen files hold `50` *Harry Potter* passages. G1 PASS: Llama-3.1-70B, single draw at
`L=100` and temperature `0.7`, leaks on `33` of `50` (mean near-verbatim recall `0.3855`), so the screen at `0.7` has
power and its zeros may be quoted.

| reading | value | registered | verdict |
|---|---|---|---|
| V1 TinyComma, `L = 20, 100, 200` | `0`, `0`, `0` of `50` | all zero | **right** |
| V1 Comma-7B, `L = 20, 100, 200` | `0`, `0`, `0` of `50` | all zero | **right** |
| V2 TinyComma `S_w`, `0.7` against `1.0` | `179.14` against `159.83` | above | **right** |
| V2 Comma-7B `S_w`, `0.7` against `1.0` (`50` of its own tokens) | `135.09` against `121.70` | above | **right** |

**Descriptive, no band.** V3: `log 64` reaches `S(x)` on `36.2%` of TriviaQA questions under the tempered TinyComma
against `35.4%` at `1.0` (median `S(x)` `5.78` against `5.86` nats). The `70`B leaks on more passages at `0.7` than
the `25` of `50` the ladder records for it at `L=100` and `1.0`, so the screen is not weaker at the deployed
temperature.

**Manuscript, as registered.** The vetting paragraph states that every rung of Table `tab:vetladder` drew at `1.0`
(and the caption says so), reports the `0.7` rungs with their control, and gives both `S_w` shifts; the appendix
TriviaQA paragraph carries `36.2%` and the meter's `83.6%` at `0.7` beside `35.4%` and `89.6%` at `1.0`. The body's
Limitations sentence, which had no room for both, now quotes the `0.7` pair, the temperature those runs sampled at
(the parenthetical form pushed two lines of the Conclusion onto page 10). The Ethics statement's prefix schedule
adds the sampling temperature. Guarded by `tests/test_v12_skipped_items.py::test_vetting_at_the_deployed_temperature_is_quoted_from_its_csv`
(eleven mutations, eleven fire).
