# Session handoff

## Current Objective

Plan v5 on branch `iclr-2027`, targeting **ICLR 2027** (abstract Sep 18, paper Sep 25). The
manuscript `~/sub/satml/iclr_2027.tex` is complete and verified: main text **exactly 9 of 9 pages**
(`Ethics` at char 264 of `pdftotext` page 10), 33 pages total, **0 overfull, 0 `??`**, 1490 numeric
literals audited with one expected miss (`64256`, the Comma-7B padded embedding count). **221 tests**
green. **feat-035..083 `done`; nothing in progress.** Compute 129.6 GPU-hours.

## What landed this session

**feat-083 — the onset residue is not Proposition 2's burstiness, and `s(x)` beats `k_crit` as a
unit by `3.8x`.** Pre-registered with a refuting band, a predicted sign and three excluded
alternative statistics *before the quantity was computed* — it costs no GPU and thirty seconds, so
nothing but the commit separates a hypothesis from a story. Spearman between each pair's
`k_crit/s(x)` and its onset ratio is **`+0.036`** at exact `p = 0.96`; the two burstiest pairs by a
factor of three sit in the middle of the coarse family and the two above `1` are the second and
third *least* bursty, so it fails at its extremes rather than for want of power. The corollary is
positive and new: `CV(onset/k_crit) = 0.434` against `CV(onset/s(x)) = 0.115`, so the running
maximum is the wrong unit for where leakage *begins* even though it is the right one for when a work
becomes *reproducible*. `collapse_robustness.csv` could not say this — it lists `k_crit` with `nan`
at `n=0`, because the `k/k_crit` overlap window is empty on these grids.

**feat-082 — the onset on a corpus the law has never seen.** The paper's first limitation, and the
plan's own risk list, is that everything runs on sixteen English genre novels. feat-080 answered
that for the order results; this answers it for the **onset**, which is the positive contribution.
`--corpus-file` is now additive on `composition_attack.py` and `onset_theory.py`; three anchors with
a Gutenberg memoriser were swept on the same 12-point grid with both baselines, on 600 excerpts of
50 public-domain books. Predictions, bands, grid and entry gate committed at `76880b9` **before any
of them decoded a token**, narrowed twice more while the runs were in flight and nothing scored.

```
pair            k=-1   onset   bracket     ratio  95% CI        no-x   Eq.(req)  pred/meas   on the novels
KL3M-520M      0.578   2.608  (2.5,2.7]   1.102  [1.07,1.42]   0.0%    2.218      0.851      1.053 [1.02,1.24]
Pleias-1.2B    0.517   2.513  (2.5,2.7]   0.895  [0.85,1.17]   0.0%    2.463      0.980      0.878 [0.79,0.96]
Phi-3.5-mini   0.270   2.704  (2.7,2.9]   0.949  [0.79,1.33]   0.5%    2.813      1.040      0.926 [0.80,1.09]
```

- **The central split reproduces.** KL3M-520M, the fine-tokenizer pair, is again the only one above
  `1` and its interval again excludes `1`; both coarse pairs are again below; every ratio lands
  within `0.05` of its twin. *Leakage beginning after the certificate has gone vacuous is a property
  of the pair, not of those novels.*
- **The level of Eq. (req) transfers; its direction still does not.** `pred/meas` `0.851`–`1.040`,
  inside the committed `[0.85, 1.15]`. The three-pair direction test returns exact `p = 1.000` (the
  floor at `n = 3`, written down as such beforehand) and the seven-pair inversion stands. **P2
  missed on all three** — the onset lands above the median of `r(x)`, where on the novels it sat at
  `q25`.

**The grid-ceiling rule, applied to the main onset table for the first time.** Appendix E commits to
extending a grid whenever the bootstrap no-crossing fraction rises materially above the others, and
the rule had only ever been applied to the seed arms. KL3M-1.7B sat at `4.3%` on a grid topping out
at `3.2` with a bootstrap upper end of `3.126`. Extended to `{3.5, 4.0, 5.0}`: no-crossing
`4.3% -> 0.0%`, onset **unmoved** at `2.578`, interval widens **upward only**. The lower end does not
move, so "both KL3M intervals exclude 1" is unaffected.

**Read-through of the main text against the CSVs — eleven corrections.** Six of Appendix D's 72
cells were one off from double rounding (`window_factor` now 6 s.f., with a test over all 72); the
anchor rate is *tied* for worst predictor at `alpha=8`, since `s_s - s_r` ranks the twelve pairs
identically; Section 2's utility gain and optimal-policy cost were endpoints quoted as ranges; the
overhead ratio is not monotone (minimum `2237` at `k=10`); Appendix E promised a no-crossing column
it did not have, carried five stale seed-words that disagreed with Section 4 on the same arms, and
one wrong CI; and `lcs_word` is the longest common **substring** in words, not subsequence.

**feat-079/080/081 registered** with evidence, `progress.md` blocks and `README_artifact.md`
sections. `compute_hours.py` now detects a fine-tune from the `[ft]` lines in a job's own log rather
than the job's name (a 10-hour undercount), and `build_artifact.sh` no longer ships
`data/gutenberg/` — 44 MB an earlier build had committed, against the README's own statement.

## Files Changed

New: `analysis/onset_gutenberg.py`, `tests/test_onset_gutenberg.py`, `tests/test_order_law_table.py`,
`results/{onset_gutenberg.csv,onset_theory_gutenberg{,_per_work}.csv,onset_theory_pairs_gutenberg.tsv,
onset_prediction_gutenberg.md}`. Modified: `analysis/{composition_attack,onset_theory,compute_hours,
order_law,order_predictors,order_crossings}.py`, `tests/{test_compute_hours,test_order_seed}.py`,
`scripts/build_artifact.sh`, `init.sh`, `feature_list.json`, `progress.md`, `README_artifact.md`,
`AGENTS.md`, `results/{onset,onset_ci,onset_table,onset_units,collapse_robustness,seed_effect,
order_law,order_predictors*,compute_hours*}.csv`, both manifests, `artifact/`. In `~/sub/satml`:
`iclr_2027.tex`, `sections/{onset,frontier,appendix_robustness,appendix_seed,appendix_limitations}.tex`.

## Recommended Next Step

1. **The appendix read-through is done** (two further corrections: a Table 3 caption that
   described a ratio of medians as a median, and "30 targets" where one of six cells is 12).
   Every table in the compiled document has now been checked against its own CSV mechanically.
2. **Then stop adding.** Every pre-registered band in `results/onset_prediction_*.md` has been
   scored, the three robustness axes are probed, and both corpora agree. Running more anchors now
   would be choosing when to stop after seeing the answer.
3. Sep 18 is abstract registration, which is `feat-016` — **human-only, never to be started.**

## Standing constraints worth re-reading before touching anything

`AGENTS.md` in full, and in particular: never push to a remote; `feat-016` is human-only; do not
modify `~/sub/neurips_2026.tex`, `output.zip`, or the committed prompt sets under `data/` (the one
writable path there is `data/gutenberg/`); nothing in `~/sub/satml/` or the artifact may identify the
authors, the sole exception being third-person `\cite{vijayavallabh2026audit}` as "an earlier audit";
`HF_TOKEN` returns 401, so run local jobs with `HF_HUB_OFFLINE=1`; ask before any **new** gated
download; never GPU 3; always `CUDA_DEVICE_ORDER=PCI_BUS_ID`; the manuscript tree sits inside a stray
home git repo — always run git with an explicit path into `DA5001_Project`; after any manuscript
edit recompile and check exit status, 0 `??`, 0 overfull, <= 9 pages of main text, and remember that
an addition to the main text costs about three times its own length in reflow; a budget violation is
per-trajectory; `master` holds the verified SaTML paper at `dd7e801` as the fallback.
