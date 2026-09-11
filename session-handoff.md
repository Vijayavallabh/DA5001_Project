# Session Handoff

**Date:** 2026-09-11 (late evening) · **Branch:** `iclr-2027` · **Target:** ICLR 2027, abstract Sep 18,
paper Sep 25. `master` holds the verified SaTML fallback at `dd7e801`.

## Current Objective

None open. feat-035..086 are `done` except the optional feat-010/011; feat-012 is superseded by
feat-024, feat-060 was wrongly withdrawn and reinstated the same day (it is `done`), feat-016 is
human-only and must never be started.

The last thread was **feat-086**, which turned the paper's largest stated limitation into an
intervention. The onset ratio falls with the words a fixed 20-token seed buys the adversary, but
seed words is 20x characters per token by construction, so the nine-pair ranking is observational.
Hand every adversary the same 13.6--15.0 **words** instead and the spread in onset/s(x) goes
**0.289 -> 0.113** (cv 9.6% -> 4.2%), `S_match/S_20 = 0.392` against a `<= 0.5` band committed before
either new arm was swept; the near-verbatim metric gives 0.399 on the same arms. Both blind arms
(open-calm at `--seed-tokens 30`) land inside their committed [0.85, 0.96], the 1B at 0.959 on the
edge. Every pair that moved moved **down**; the five that did not move were already at the matched
context, which makes them the control rather than the effect. 61% of the spread is the benchmark's
seed convention, 39% is not, and the residue is reported as a residue.

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

## Addendum, 2026-09-11 late evening

**feat-086 is done and the read-through it ran alongside found five defects, all now pinned by
tests.** The one worth carrying forward: **a test that returns early when a file is missing can pass
by never running.** `tests/test_order_law_table.py` and `tests/test_compute_hours.py` both guarded
on `os.path.expanduser("~/sub/satml/...")`, and `~` is not the project home on this box, so the
72-cell appendix check AGENTS.md advertises had never executed. Manuscript paths in tests now go
through `tests/manuscript.py` (`$SATML_DIR`, else `../sub/satml`). The others: the seed-effect
figure plotted seven hardcoded word counts the manuscript had already corrected; four numbers in
Appendix C's prose were two pair-counts stale and survived `audit_numbers.py`, which only asks
whether a literal appears in *some* CSV; "predicts every other arm to within 5.1%" was a mean, not a
bound; and `compute_hours.py` did not scan `output/logs/`, so a day of sweeps left the total
unmoved.

**Verified after all of it.** `./init.sh` green, **238 tests**. Manuscript: tectonic exit 0, 0
overfull, 0 `??`, **35 pages**, main text 9 of 9 with no body prose on `pdftotext` page 10.
`audit_numbers.py`: 1667 literals, one expected miss. Compute **133.4 GPU-hours**, fine-tune share
<= 27.8, and the LLM-usage sentence says 133 / 28. Artifact rebuilt, 601 files.

**Recommended next step, unchanged in kind:** another pass of the same mechanical read-through. The
two table checks now run (`tests/test_order_law_table.py`, `tests/test_onset_table.py`), so the next
thing to mechanise is Appendix E's seed table and the frontier tables, which are still read by eye.
