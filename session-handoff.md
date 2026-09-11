# Session Handoff

**Date:** 2026-09-11 (evening) · **Branch:** `iclr-2027` · **Target:** ICLR 2027, abstract Sep 18,
paper Sep 25. `master` holds the verified SaTML fallback at `dd7e801`.

## Current Objective

None open. feat-035..085 are `done` except the optional feat-010/011; feat-012 is superseded by
feat-024, feat-060 is withdrawn, feat-016 is human-only and must never be started.

The last thread was **feat-085**, the contingent control the feat-084 pre-registration committed
before the eighth pair was swept. The eighth pair (`open-calm-1b`, the only one of 21 surveyed
tokenizers inside the 2.4--3.4 characters-per-token gap) entered the onset analysis on the weakest
memoriser in the set, which was the one pre-registered confound that fired. `open-calm-3b` shares
its tokenizer exactly, so granularity, the 9.5-word seed, `s(x)` (+0.5%) and burstiness (1.541 vs
1.542) are held fixed and only the scale and the memoriser change. It entered on a sampled `k = -1`
recall of **0.924** -- the strongest in the set, 5.1x the eighth pair's -- and landed at
**onset/s(x) = 0.9933, [0.959, 1.075]**, inside the committed interpolation band [0.927, 1.052].
Memoriser strength was not what placed the eighth pair between the clusters.

At nine pairs: seed-words Spearman **-0.958** (exact p = 0.0002, from -0.946), matched-context
subgroup exact **p = 0.008** over 126 subsets (from 0.018), collapse spread unchanged at 0.027,
`s(x)` still the best of four normalisers (cv 10.2% against 15.0 / 23.8 / 37.7). Nothing reverses.

## Files Changed

- `results/onset_prediction_granularity_gap.md` -- the 3B scored against its committed bands.
- `results/*.csv` -- the whole nine-pair chain re-run (`onset`, `onset_table`, `onset_units`,
  `collapse_robustness`, `onset_ladder`, `onset_burstiness`, `seed_effect`, `natural_pair`,
  `surprisal_cdf`, `score_predictions`, `compute_hours`), figures rebuilt and copied.
- `analysis/onset_gutenberg.py` -- `spearman` now imports the tie-aware average-rank version from
  `analysis/seed_effect.py`; the value-keyed one collapsed ties onto a single rank. Both CSVs it
  produces are byte-identical after the change.
- `analysis/seed_effect.py` -- `onset_seed_words.csv` stores seed words at 4 dp, not 1: at 1 dp
  Pleias-350M (13.86) and Phi-3.5-mini (13.94) tie and the paper's correlation cannot be recomputed
  from its own CSV.
- `tests/test_granularity_gap.py` (new, 4 tests) -- the committed bands, both open-calm ratios
  inside the interpolation band, the matched contrast, and the correlation against the CSV. **228
  tests.**
- `feature_list.json` -- feat-085 registered `done` with its evidence. 80 features.
- `~/sub/satml/` -- eight to nine pairs throughout: `iclr_2027.tex` (abstract, compute figures),
  `sections/iclr_intro.tex`, `sections/onset.tex` (table row, counts, correlation, the control),
  `sections/iclr_closing.tex` (compressed ~4 lines), `sections/appendix_robustness.tex`,
  `sections/appendix_seed.tex`, `sections/appendix_limitations.tex`.
- `AGENTS.md`, `init.sh`, `README_artifact.md`, `progress.md`, `artifact/`.

## Verified

`./init.sh` green, 228 tests. Manuscript: tectonic exit 0, **0 overfull, 0 `??`, 33 pages, main text
9 of 9 with no body prose on `pdftotext` page 10 at all**. `analysis/audit_numbers.py`: 1582 math
literals, one expected miss (`64256`). Compute **132.7 GPU-hours**, fine-tune share <= 27.8.

## Recommended Next Step

Nothing is blocked and no experiment is half-run. The paper is complete at nine pairs. The useful
work left is another end-to-end read-through of the compiled PDF against `results/*.csv` -- the last
one found thirteen corrections -- and then the human-only feat-016 steps, which the agent must never
start.

**One measurement caution replaces an older one.** The page budget is now checked by reading
`pdftotext` page 10 and confirming it carries the running header, the line-number gutter and
`Ethics Statement` and **no body prose**. The `char 264` rule this file and AGENTS.md carried until
today did not detect two lines of the Conclusion spilling onto page 10; the character index is not
monotone in the spill, because pdftotext emits the gutter and the text column separately.
