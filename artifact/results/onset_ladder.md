# The graded-memoriser ladder: the sharpest available test of the derivation

## Why

The onset law says leakage begins at `r(x) = s_s(x) - s_r(x)`. The competing reading is that
leakage begins at a fixed fraction of `s_s(x)` alone -- the "0.89" both existing pairs happen to
share. Across *different* pairs the two are hard to separate, because changing the anchor changes
`s_s` and `s_r` together and a strong memoriser always has small `s_r`.

The ladder removes that confound. **One anchor, one base model, one corpus; only training strength
varies.** So `s_s(x)` is identical across rungs by construction, and `s_r(x)` is the only thing
that moves.

- Constant-coefficient hypothesis: every rung has the same `s_s`, so every rung onsets at the
  **same budget**.
- Eq. (req): the onset is `s_s - s_r`, so it must **fall as memorisation strengthens** and rise as
  it weakens.

These predictions differ in *sign of the trend*, not just in magnitude, which is the strongest form
this test can take. A flat ladder falsifies the derivation outright.

## Design

Base and anchor: `PleIAs/Pleias-350m-Preview`, self-paired. Rungs are the same fine-tune stopped at
different training losses (deterministic seed, identical data order), which is why they differ only
in how thoroughly the passages were learned:

| rung | stop loss | expected role |
|---|---|---|
| A | 0.20 | weakest still-admissible memoriser |
| B | 0.10 | |
| C | 0.05 | |
| D | 0.03 | already trained: sampled recall 0.901, s_r = 0.326 |

Every rung must clear the admissibility bar (sampled `k=-1` recall >= 0.10) to enter the analysis;
the fine-tune recipe prints the verdict. A rung that fails is reported as failed, not dropped
silently -- an earlier 12-epoch/rank-64 memoriser of this same base reached only 0.022 and is
recorded in `onset_prediction_pleias350m.md`.

## Protocol

1. Train rungs A-C (`--stop-loss` as above, everything else identical to D).
2. For each admissible rung, `analysis/onset_theory.py` gives `s_r` and the predicted onset.
   **Record the predictions before sweeping.**
3. Sweep each rung on the same k grid, same 100 passages, same seeds, with the mandatory `k=-1`
   and `k=0` baselines.
4. Report measured onset against both hypotheses.

---

## The ladder does not do what it was designed to do, and the predictions say so before any sweep

Computing each rung's prediction *before* sweeping (which is why the predictions are computed first)
shows the design is weaker than intended:

| pair | s_r | derivation, q25 | constant 0.888*s_s | gap |
|---|---|---|---|---|
| TinyComma + mem. Llama-8B | 0.194 | 2.86 | 2.88 | 0.02 |
| Comma-7B + mem. Comma-7B | 0.179 | 2.13 | 2.13 | -0.00 |
| Pleias-350M + mem. Pleias-350M | 0.326 | 2.96 | 3.16 | **0.19** |
| ladder rung stop-loss 0.10 | 0.374 | 2.94 | 3.16 | **0.22** |
| ladder rung stop-loss 0.20 | 0.411 | 2.90 | 3.16 | **0.25** |

**The between-rung trend is unmeasurable.** Training strength moves sampled recall 4.5x (0.202 to
0.901) but moves `s_r` only 0.411 -> 0.326, so the predicted onset moves **0.06 nats** across the
whole ladder, against a grid resolution of 0.10. The sign-of-trend test the ladder was built for
cannot be run.

The reason is a genuine tension, not an accident of these settings. Admissibility needs a memoriser
strong enough that the unconstrained model reproduces the work, and any such memoriser has small
`s_r`. The usable window in training loss is roughly [0.03, 0.25] -- below it nothing more is
learned, above it extraction collapses (a model at loss ~0.35 gave sampled recall 0.022) -- and that
window maps to a narrow `s_r` band.

**What the ladder is still worth.** Each rung independently tests the same 0.19-0.25 nat gap between
the two hypotheses, with the anchor held fixed, so the rungs are *replications* rather than a trend.
That is worth having but it is not decisive on its own.

**What actually moves `s_r`: model size, not training strength.** `s_r` is 0.18 at 7B, 0.19 at 8B and
0.33 at 350M -- a 1.8x range against the ladder's 1.26x. Pairs across model scales are therefore the
discriminating lever, which is what the remaining self-paired memorisers supply. Pair 3 alone already
carries a 0.19 nat gap between the hypotheses, and its sweep is the first real test.
