# ROUGE-L threshold sensitivity for the non-literal leakage event

Derived recount over committed per-passage CSVs; no generation. Written and committed before the
script was run; nothing above `## Scoring log` is edited afterwards. Descriptive (`_note.md`).

Produced by `.venv/bin/python analysis/rouge_threshold.py --out results` ->
`results/rouge_threshold.csv`.

## Why

The paper's non-literal event is ROUGE-L `>= 0.5`. `onset_prediction_memfree_headtohead.md`
excluded in advance "scoring the paraphrase event with a threshold other than the committed
ROUGE-L >= 0.5", and that stays true: **this is a sensitivity table beside the registered
verdicts, never a replacement for them.** Referees asked whether the `0` of `100` against `47` of
`100` contrast depends on where the line is drawn.

## What is counted

Passages (of 100) at or above `theta in {0.3, 0.4, 0.5, 0.6, 0.7}`, per source column, from:

* **Anchor pools, clean anchor** (`selfix_clean_grid64`, `selfix_clean_n256`,
  `selfix_clean_paraphrase`, `selfix_clean_multilingual`): `anchor_max_rouge` (best of the whole
  pool -- selector-free), `rouge_n{n}` (the memoriser-likelihood selector's pick) and
  `oracle_rouge_n{n}`.
* **Contaminated anchors** (`selfixR_*`, `selfix256_*`): the same columns, labelled contaminated.
* **Memoriser alone** (`risky_alone_rouge`, the `k=-1` baseline in each file).
* **Blocklist / TokenSwap decoders** (`blocklist_decode_per_passage*.csv`): `rouge_l` per `arm`
  (`-1` is that file's rule-off control).

Values are the stored 4-decimal ROUGE-L, compared with `>=`. For a zero count the one-sided 95%
upper bound on the per-passage rate, `1 - 0.05^(1/100) = 2.95%`, is quoted.

## Expected reading, fixed in advance

Clean-anchor pool maxima stay at `0` for `theta >= 0.5` and may become non-zero at `0.3`
(the stored `rouge_ge_0p3_pct` already reads `0.0` for the served picks at TinyComma, so a
non-zero there would come from the pool maximum only); the memoriser falls monotonically from
`71` at `0.3` through `47` at `0.5`.

## Scoring log

### Scored 2026-09-24 14:11 IST --- the contrast does not depend on where the line is drawn

`.venv/bin/python analysis/rouge_threshold.py --out results` -> `results/rouge_threshold.csv`
(1,510 rows).

- **Clean anchors: `0` of `100` at every `theta` from `0.3` to `0.7`**, in all four clean files
  (`grid64`, `n256`, `paraphrase`, `multilingual`), for the pool maximum (selector-free), for every
  served pick and for every oracle pick (17 columns). The one-sided 95% bound on the per-passage rate
  is `2.95%` at every threshold. The expected reading allowed a non-zero pool maximum at `0.3`; there
  is none.
- **The memoriser alone** falls monotonically with `theta` in every file: `71`--`93` at `0.3`,
  `47`--`83` at `0.5` (the `47` is the paper's figure, `grid64`), `39`--`56` at `0.7`.
- **Contaminated anchors** (24 files) are the informative case: the pool maximum clears `0.5` on
  `84`--`100` passages and `0.3` on `98`--`100`; above `0.5` it ranges `0`--`100`, which is the anchor
  and not the threshold.

The registered `0.5` verdicts are unchanged; this table sits beside them.
