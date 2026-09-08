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
