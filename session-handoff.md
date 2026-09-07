# Session handoff — 2026-09-07, branch `iclr-2027`

## Current Objective

Plan v4 (`.claude-private/plans/radiant-stargazing-newell.md`): turn the SaTML audit of one
mechanism into a frontier theorem for the **class** of bucket-metered decoders, for **ICLR 2027 —
abstract Sep 18, paper Sep 25**. The user reversed the venue decision and accepted that the finished
SaTML submission is abandoned. `master` holds it at `dd7e801` as the fallback; do not delete it.

## What exists now

Phase 4 features feat-035..040 are `done` with evidence in `feature_list.json`. 67 tests,
`./init.sh` passes, ~5 GPU-hours spent.

| Claim | Where | Status |
|---|---|---|
| Vacuity threshold is $K=S(x)$ for **every** Rényi order | `sections/frontier.tex` Prop. 1 | proved; conversion bound checked on 40,000 random pairs, tight to 1e-15 |
| $k_{\rm crit}\ge s(x)$, so protection outruns certification | `sections/frontier.tex` Prop. 2 | proved; pinned by `tests/test_regimes.py` |
| The margin $s(x)/c_{\rm use}$ **rises** with safe-model capability | `results/anchor_scaling_summary.csv`, `_paired.csv` | 10 models, 3 corpora; 16/16 novels up per family, sign test p=3.05e-05 |
| The uncertified interval is an **opening** effect | `results/opening_effect*.csv` | 4 anchors; binds at token 0 in 87.7–90.5%; 4.25–6.10x whole-work, 1.32–1.48x with 1 prefix token; denominator-invariant |
| The realised price is not a model property | `results/budget_drift.csv` | 5–32x gap, 6 classes x 2 budgets, CPU only |
| Same budget + same vacuous certificate, different protection | `results/renyi_sweep.csv` | at k=3, 100% vacuous under both orders, recall 0.097 (α=1) vs 0.054 (α=2) |
| A second complete 7B (anchor, risky) pair decodes | `output/phase4/memorizing_comma7b` | nv-recall 0.915 train, **0.000** held-out test |

## Three things that override older text

1. **The SaTML abstract's anchor-scaling inference is wrong.** $s(x)$ does fall with anchor scale
   (0.778 → 0.685 nats/char) but $c_{\rm use}$ falls further (0.191 → 0.137), so the 7B anchor has
   the **largest** margin of ten, not the smallest. Do not reuse that sentence.
2. **feat-028b was never impossible.** The blocker was a guard demanding
   `embedding rows == len(tokenizer)`; comma-7b pads 64,000 tokens to 64,256 rows. Fixed in
   `a_patch/factory.py` with `_mask_pad_rows`; the Llama-3 path is a verified no-op.
3. **Never quote the 6.10x uncertified interval alone.** One token of genuine prefix collapses it
   to 1.36x, and the oracle-window adversary supplies exactly that.

## Recommended next step

1. **Finish the two running sweeps** and merge them:
   `output/phase4/renyi_{1_0,2,4,8}` → `.venv/bin/python analysis/renyi_sweep.py --out results`;
   `output/phase4/comp_comma7b` is the second-anchor validation of the frontier — compare its
   measured reproduction onset against `analysis/regimes.py --model common-pile/comma-v0.1-2t`.
2. **Write the remaining sections.** `iclr_2027.tex` compiles at 4 pages, 0 overfull, with
   `frontier.tex`, `scaling.tex` and three figures. Missing and referenced (3 `??`):
   `sec:utility`, `sec:attack-results`, and `prop:path` (Prop. 4, to be lifted from
   `sections/appendix_theory.tex`). **9 pages of main text is the hard ICLR limit**; references and
   appendices are free. An **AI use statement is required**.
3. **Do not** register the abstract or submit — `feat-016` is human-only.

## Files changed this session

New: `analysis/{regimes,anchor_scaling,budget_drift,opening_effect,renyi_sweep}.py`,
`a_patch/renyi.py`, `scripts/download_safe_models.py`, `figures/make_figures_v4.py`,
`tests/test_{regimes,padded_vocab,renyi}.py`,
`~/sub/satml/{iclr_2027.tex,sections/frontier.tex,sections/scaling.tex,iclr2027_conference.sty,...}`.
Modified: `a_patch/factory.py` (padded vocab + Rényi), `recipes/finetune_memorizing.py` (`--no-chat`),
`dap/e1.py` and `analysis/composition_attack.py` (`--constraint` validator), `AGENTS.md`,
`progress.md`, `feature_list.json`.

## Blockers / risks

- **18 days.** If the theorem does not survive, fall back to `master` and submit to SaTML Sep 29.
- The utility boundary of the theorem is **measured, not proved**; two candidate proofs were
  falsified against the logs and the reasons are recorded in `progress.md`. Do not re-derive them.
- `c_use` is measured against one risky model (Llama-3.1-8B-Instruct). The cross-safe-model trend is
  internally valid, but a second risky model would strengthen the rebuttal.
