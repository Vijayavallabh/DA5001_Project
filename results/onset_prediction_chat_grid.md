# Pre-registration: where the chat-template meter overtakes selection (feat-196)

**feat-196.** Committed **2026-09-24**, before any token of the new arms is decoded. Nothing above
`## Scoring log` is edited after the first judge call.

## Why

Table 3's chat-template block has two budgets: at `k=1` selection leads, `+0.0410 [+0.012, +0.0705]`,
and at `k=10` the meter wins by `0.138`. A referee report (2026-09-24, sixth round) calls the
crossover between them "the most deployment-relevant number the paper could report" and asks for
`k in {0.5, 2, 3, 5}`. `k=0.5` already exists (`output/sweep_chat`, the same run as the `k=1` row);
`k = 2, 3, 5` are generated here exactly as feat-184 generated `k=10`.

## What runs

`h1.py --use-chat-template --k-values 2 3 5 --trajectories-per-prompt 1 --seeds 52 53 54
--cap-neutral 200 --cap-factual 150 --cap-creative 150 --cap-val 0 --cap-test 0 --cap-attack-train 0
--batch-size 48 --trust-remote-code --output-dir output/feat196/chat_grid`, temperature `1.0`, no
penalty: feat-184's command with the budgets changed.

Judging: `analysis/order_averaged_h2h.py --deecho`, judge B, seed `7717`, the chat opponent
(`output/sweep_chat`, `k=-1`), the anchor-alone control (`output/sweep_chat`, `k=0`), selection at
`n=64` (committed pool and rewards), as in feat-184's run B3.

- **J5** metered `k=0.5` (`sweep_chat`), extra `k=2`.
- **J6** metered `k=3`, extra `k=5`.

## Gates

- **G0.** Each new arm covers the `500` prompts and every trajectory stays within its budget.
- **G1.** Selection's per-prompt levels equal feat-184 B3's (same text, opponent and judge).

## Predictions

`D3` at `k=0.5` and `k=3`; `D5` (the extra arm's gain minus selection's) at `k=2` and `k=5`.

- **P1.** `k=0.5`: **CONFIRMED** (selection ahead). The meter binds hardest here and is its anchor.
- **P2.** `k=2`: **TIE**. **P3.** `k=3` and `k=5`: the meter ahead (**REFUTED** / **INCUMBENT
  WINS**). The crossover lies between `k=1` and `k=3`.
- **P4, descriptive.** The binding share and `K/S_w` at each budget. Every budget from `k = 1` up is
  vacuous for a `50`-token window (`K/S_w >= 1.25`), so wherever the crossover lands it is at a budget
  whose certificate excludes nothing; that is recorded now so it cannot be offered as a finding later.

## What the manuscript does with each outcome, fixed now

Table 3's chat block reports every budget on the grid, and the text states the smallest budget at
which the meter's interval lies above selection's and its `K/S_w`, whatever it is.

## Excluded in advance

- Budgets or seeds chosen after a judge call; pooling with other passes; level comparisons across
  passes.

## Scoring log
