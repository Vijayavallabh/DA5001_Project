# Session handoff

## Current objective
Plan v5, branch `iclr-2027`, targeting ICLR 2027 (abstract Sep 18, paper Sep 25). The manuscript is
complete and verified; the remaining work is whatever the plan opens next.

## What happened this session
1. **feat-072 corrected twice.** The order-price probe's leakage column was fidelity, which is a
   bounded average and cannot see a rare event; replaced with `sum_t log p_theta(x_t)` over the
   protected tokens, whose exponential is the reproduction probability. The pre-registered sanity
   bracket then fired on its first run and found a **split bug**: every phase-5 memoriser is
   fine-tuned on `attack_train` + `val` with `test` held out, and the two splits are disjoint in
   novel, so both `order_price.py` and `marginal_price.py` were scoring "protected" text the model
   has never seen. Both now default to `attack_train`.
2. **feat-070 re-run on the memorised split** and published as a third target type rather than
   replacing the held-out one. Gain ratio 1.031-1.133 across 2 pairs x 3 targets x 4 budgets; the
   manuscript's "property of the geometry" sentence is now supported instead of asserted.
3. **feat-073 (new): the order comparison at matched utility.** `analysis/order_frontier.py` sweeps
   a 12-point `k` grid, finds the budget at which each order buys exactly what the audited decoder
   buys at the published one, and reads the rare-event functional there. At `k=3` the ranking
   reverses on KL3M-520M and every matched budget is past that pair's vacuity threshold; at `k=1`
   the higher order dominates by 871x to 3.3e7 depending on the pair, non-monotone in alpha. This
   closes the concession in Appendix D with an answer rather than an admission.
4. `seed_effect.seed_words` moved onto the split the sweeps actually used; seed words 7.3 -> 7.5 and
   13.7 -> 14.6, every ratio and interval unchanged.
5. **feat-074 (new): what predicts the order's value? Nothing measured does.** Two hypotheses were
   pre-registered and both refuted --- the audited decoder's distance from its own fidelity ceiling
   (`F`, refuted at 2.1-7.7 decades between pairs at matched `F`), and the memoriser's per-token
   log-probability, whose three-pair ordering was tested out of sample on a fourth and failed
   (Spearman `+0.800`, exact two-sided `p = 0.33`, predicted `+1.000`) and was deleted rather than
   re-fitted. Four further candidates were scored and none reaches `|0.4|`. What survives is a
   stable pair effect spanning **5.4 decades** at `k=1` and changing sign at `k=3` on two of four
   pairs. Phi-3.5-mini and Comma-7B added; Comma needed `--dtype bfloat16` to share a card, with a
   bfloat16-against-float32 control on KL3M-520M measuring the cost at <= 0.78 nats per window.

## State
- **192 tests** (`./init.sh` green), 69 features, none in progress, feat-035..074 `done`.
- Manuscript: main text **exactly 9 of 9 pages** (Ethics at char 264 of page 10, i.e. the body ends
  at the foot of page 9), 30 total, 0 overfull, 0 `??`, 1230 numeric literals audited with 1
  expected miss (`64256`). Compute figure 111 GPU-hours. The abstract carries the matched-utility
  result, swapped in for a sentence of equal length: adding four lines to the abstract cost
  **thirteen** lines of reflow further down, so any abstract edit must be length-neutral.
- Artifact rebuilt: 531 files, `artifact.zip` 27M.

## Recommended next step
feat-072/073/074 are complete and in the paper (Appendix~\ref{app:matched}, the closing paragraph of
Section 6, and one clause of the abstract). Three candidates for what comes next, in order of
expected value:

1. **Three more pairs on the frontier sweep.** The pair effect is stable across three orders but
   rests on four pairs; `output/phase5/` has memorisers for KL3M-1.7B and Pleias-350M, and the
   seven-pair onset set names a seventh. Seven pairs would let "nothing measured predicts it" be
   quoted with a rank test that has real power (exact p = 1/5040 rather than 1/24). ~40 min of one
   A100 per pair.
2. **The one-corpus limitation.** Everything runs on sixteen English genre novels, and Limitations
   says so. The Gutenberg cache under `data/gutenberg/` is already scored by `anchor_scaling.py`;
   a memoriser on public-domain prose would give a second corpus for the onset and the frontier.
3. **Nothing.** The paper is verified end to end and both deadlines have slack. Stopping is a
   legitimate choice and the fallback at `dd7e801` on `master` is intact.

## Standing constraints
Never push to a remote; `feat-016` is human-only. Never commit inside `~/sub/satml` (stray home git
repo) -- always `git -C .../DA5001_Project`. GPU 3 is a 4 GB T400: never use it, and always set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. `HF_HUB_OFFLINE=1`; `meta-llama/*` stays gated. `master` holds the
verified SaTML paper at `dd7e801` as the fallback.
