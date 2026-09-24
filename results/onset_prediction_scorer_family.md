# Pre-registration: the headline with a scorer from another family (feat-198)

**feat-198.** Committed **2026-09-24**, before any candidate is scored. Nothing above `## Scoring log`
is edited after the first reward is computed.

## Why

Two referee reports (2026-09-24, sixth round) note that selection's pointwise reward is
`Qwen2.5-7B-Instruct` and that the largest judged reading of the headline, `+0.1210`, comes from
`Qwen2.5-14B`, the scorer's family. The paper answers with the judges (without the two Qwen judges,
three of four exclude zero); this arm answers with the scorer. It changes one thing: the model that
scores the `64` draws.

## What runs

- Pool: the committed headline pool, `output/phase5/sel_anchor64` (`64` draws per prompt, `500`
  prompts), unchanged.
- Scorer: `google/gemma-2-27b-it`, bf16, the committed template and path
  (`analysis/pool_rewards.py`, which calls `analysis/compute_matched.reward_cache` on the same
  `(prompt, generation)` pairs the committed Qwen cache scored) ->
  `results/selection_rewards64_gemma27b.csv`. Selection serves the argmax at `n=64`, ties by index;
  the certificate is `log 64` whatever the scorer.
- Judging: `analysis/order_averaged_h2h.py --deecho --rewards results/selection_rewards64_gemma27b.csv
  --tag scorer_gemma27b`, judge B, seed `7717`, both orders; every other input is the headline pass's
  (opponent and anchor control `output/sweep_plain`, meter `output/phase2/conc_all` at `k=10`).
- Host B, GPU 7 (idle), `scripts/run_feat198.sh`.

`gemma-2-27b-it` is judge G's checkpoint, so judge G's reading of this arm would be the scorer
grading itself; **the registered reading is judge B's alone**.

## Gates

- **G0.** The cache holds `32,000` finite rewards, `64` ranks for each of the `500` prompts; the
  judge pass covers `500` prompts.
- **G1.** The new scorer changes the served draw on at least `10%` of prompts; if it serves the
  committed pick on more than `90%`, the arm cannot distinguish the scorers and is reported as
  UNINFORMATIVE rather than as a replication.

## Predictions, on the script's own readings

- **S1.** `D3` (selection's gain minus the meter's at `k=10`) reads **CONFIRMED** (interval above
  zero), as the headline's `+0.0505 [+0.0155, +0.0860]` does.
- **S2, descriptive.** Selection's own gain `D1` lies within `0.03` of the headline's `+0.1015`.

## What the manuscript does with each outcome, fixed now

Section 4's robustness paragraph reports the arm beside the Qwen-family judge caveat, whatever it
reads: CONFIRMED says the difference does not depend on the Qwen scorer; UNRESOLVED or REFUTED says it
does, and the heading of that paragraph adds the scorer to what the difference does not survive.

## Excluded in advance

- Any other scorer, `n`, template or judge chosen after a reward is read; reading judge G on this arm.

## Scoring log
