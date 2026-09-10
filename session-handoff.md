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

## State
- **190 tests** (`./init.sh` green), 68 features, none in progress, feat-035..073 `done`.
- Manuscript: main text **exactly 9 of 9 pages**, 29 total, 0 overfull, 0 `??`, 1194 numeric
  literals audited with 1 expected miss (`64256`). Compute figure updated to 110 GPU-hours.
- Artifact rebuilt: 521 files, `artifact.zip` 27M.

## Recommended next step
The three feat-072/073 tables are in the appendix and Section 6. The obvious next question the
sweep opens: at `k=1` the order's advantage spans four orders of magnitude between two pairs, and
nothing in the paper predicts which pair gets which. A third and fourth pair (Phi-3.5-mini and
Comma-7B memorisers exist under `output/phase5/`) would say whether that spread tracks `s(x)`,
`s_r/s_s`, or nothing -- and "nothing" is itself the paper's thesis, so either answer is reportable.
Cost is about 40 minutes of one A100 per pair.

## Standing constraints
Never push to a remote; `feat-016` is human-only. Never commit inside `~/sub/satml` (stray home git
repo) -- always `git -C .../DA5001_Project`. GPU 3 is a 4 GB T400: never use it, and always set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. `HF_HUB_OFFLINE=1`; `meta-llama/*` stays gated. `master` holds the
verified SaTML paper at `dd7e801` as the fallback.
