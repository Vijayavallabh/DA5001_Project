# Pre-registration: the non-empty rule across Table 2 (feat-209)

**feat-209.** Committed **2026-09-25**, before any judge call of this arm. Nothing above `## Scoring log` is
edited after the first judge call.

## Why

Review 4 (sixth round, Q9): "Excluding empties is itself a selection rule, still certified at `log n`. ...
Why is it not the default, and do the conclusions change if every table is rerun with it?" The paper
answers the first half (the rule was fixed before the empties were seen, and switching the headline to it
afterwards would choose a rule on its outcome) and measured the headline under it on CPU for the `41`
prompts whose pick changes (`+0.056 [+0.0195, +0.093]`, `results/nonempty_rule.csv`). This arm answers the
second half on the rows of Table 2 that carry its conclusions, judged in full on one host.

## What runs

`analysis/nonempty_arms.py` writes the rule as a servable arm for each pool (the highest-reward draw whose
recovered text is non-empty, ties to the lowest index; if all `64` are empty, the committed pick): the
headline pool, the temperature-`0.7` pool and the Comma-7B pool. Each row below re-runs the row's own judge
pass (judge B, seed `7717`, both orders, `--deecho`) on host B with the non-empty arm as its extra arm
(replacing the row's own extra arm, if it had one), so the committed pick and the non-empty pick are judged
in the same pass against the same opponent. Tags `nonempty_<row>`.

| row | pass (committed command, extra arm replaced) |
|---|---|
| released `8`B, `k=10` (the headline) | default arms |
| `70`B base, `k=20` | meter and anchor `output/phase5/imit_llama70b` |
| `70`B base, `k=0.5` | the same, `k=0.5` |
| chat template, `k=10` | opponent and anchor `output/sweep_chat`, meter `output/feat184/chat_k10` |
| chat template, `k=3` | opponent and anchor `output/sweep_chat`, meter `output/feat196/chat_grid` |
| temperature `0.7`, `8`B, `k=10` | opponent, anchor and meter `output/feat195/t07_8b`; the `0.7` pool |
| temperature `0.7`, `70`B, `k=20` | opponent and anchor `output/feat195/t07_8b`, meter `output/feat195/t07_70b`; the `0.7` pool |
| AnchoredByte, `k=0.1` | Comma-7B pool as selection and anchor, meter `output/anchoredbyte/comma7b_70b` |

**Reading per row.** The committed rule's `D3` (the pass's own row) and the non-empty rule's `D3'`, the same
statistic with the non-empty pick in place of the committed one: selection's gain over its `n=1` control
minus the meter's over its anchor, paired over prompts, `10,000` resamples, `95%`, labelled CONFIRMED /
UNRESOLVED / REFUTED by the side of zero its interval lies on.

## Gates

- **G0.** Every pass covers the `500` prompts; the arm builder asserts that a pick changes only where the
  committed pick is empty.

## Predictions

- **N1.** No row's label differs between `D3` and `D3'`. *Predicted: none of the eight changes.*
- **N2.** The headline row's `D3'` is CONFIRMED. *Predicted: CONFIRMED.*

## What the manuscript does with each outcome, fixed now

Appendix `app:empties` reports every row's `D3` and `D3'`. Section 4's sentence on the non-empty rule says
whether any row of Table 2 changes its reading under it, naming the rows that do.

## Excluded in advance

- Any other row, judge, seed or rule chosen after a judge call; judging on another host.

## Scoring log

### Scored 2026-09-25 12:49 IST --- N1 right (no row changes its label), N2 right (the headline's `D3'` is CONFIRMED)

All jobs exited `0` on host B, GPU 4 (`scripts/run_feat209.sh`, times IST): the three arms `12:22`-`12:25`, the eight
judge passes `12:25`-`12:44`. Scored by `.venv/bin/python analysis/nonempty_tables.py --out results` ->
`results/nonempty_tables.csv`.

**Gate.** G0 PASS: every pass holds `500` per-prompt rows, and the arm builder's assertion (a pick changes only where
the committed pick is empty) held for all three pools: `41`, `14` and `30` picks changed in the headline,
temperature-`0.7` and Comma-7B pools.

| row | `D3`, committed rule | `D3'`, non-empty rule | labels |
|---|---|---|---|
| released `8`B, `k=10` (headline) | `+0.059 [+0.0235, +0.094]` | `+0.058 [+0.0225, +0.093]` | CONFIRMED, CONFIRMED |
| `70`B base, `k=20` | `+0.0815 [+0.0485, +0.1155]` | `+0.0805 [+0.0475, +0.113]` | CONFIRMED, CONFIRMED |
| `70`B base, `k=0.5` | `+0.1085 [+0.078, +0.1395]` | `+0.1075 [+0.0765, +0.1375]` | CONFIRMED, CONFIRMED |
| chat template, `k=10` | `-0.145 [-0.184, -0.1065]` | `-0.1485 [-0.187, -0.11]` | REFUTED, REFUTED |
| chat template, `k=3` | `-0.051 [-0.0885, -0.0135]` | `-0.0545 [-0.091, -0.0175]` | REFUTED, REFUTED |
| temperature `0.7`, `8`B, `k=10` | `-0.081 [-0.113, -0.049]` | `-0.0825 [-0.114, -0.051]` | REFUTED, REFUTED |
| temperature `0.7`, `70`B, `k=20` | `+0.0485 [+0.0145, +0.082]` | `+0.047 [+0.014, +0.081]` | CONFIRMED, CONFIRMED |
| AnchoredByte, `k=0.1` | `+0.1385 [+0.1155, +0.162]` | `+0.1475 [+0.125, +0.1705]` | CONFIRMED, CONFIRMED |

N1: `8` of `8` rows keep their label (**right**). N2: the headline's `D3'` is CONFIRMED (**right**). The non-empty
pick moves the judged level on `28`, `24`, `6` and `21` prompts in the four pools' passes, and moves `D3` by at most
`0.009` in any row.

**Post hoc, descriptive, no band** (columns appended after every registered bootstrap; the registered columns were
checked unchanged). The committed rule's `D3` in this pass against the same row in the pass Table 2 was built from:
identical in the four rows whose committed passes were judged on host B (chat `k=3` and both temperature-`0.7` rows,
per `analysis/served_opponent.py`; AnchoredByte, per `results/onset_prediction_anchoredbyte.md`) and shifted by
`+0.0085`, `-0.001`, `+0.0035` and `+0.0035` in the four judged on a local A100 (headline and chat `k=10`, feat-184,
per `results/onset_prediction_chat_grid.md`; the two `70`B rows, feat-185, queued behind it). Order averaging draws
nothing, so a shift is a difference in how the judge ran (machine or batch composition), not a re-roll; no label
moves.

**Manuscript, as registered.** Appendix `app:empties` reports every row's `D3` and `D3'`; Section 4's sentence on the
non-empty rule says that no re-judged row of Table 2 changes its reading under it.
