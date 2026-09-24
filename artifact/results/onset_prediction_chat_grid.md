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

## Addendum before launch, 2026-09-24 23:15 IST (no token decoded, no judge called)

- **Host.** Every local card is held by another job of this account (the first launch died at load,
  `output/logs/feat195_failed_launch_2209/`, `output/feat196/` empty), so the new budgets and both
  judge passes run on host B's GPU 7, with the committed `output/sweep_chat` copied there unchanged.
- **G1 needs a same-host reference.** G1 as written compares against feat-184's pass, which was judged
  on a local A100. A greedy bf16 judge is deterministic on one host (each arm is judged alone, eight
  items at a time in prompt order), but it is not bit-reproducible across hosts (caution (as)), so G1
  as written cannot tell a pipeline defect from host arithmetic. The reference is therefore fixed now,
  before J5 or J6 runs: feat-184 B3's own command (`--baseline-dir output/sweep_chat --metered-dir
  output/sweep_chat --k 1 --anchor-dir output/sweep_chat`) re-run on host B as `--tag
  served_k1chat_hostB`, on committed text only. **G1, restated: selection's per-prompt levels in J5
  and J6 equal that reference's on all `500` prompts.** How far the host-B reference agrees with
  feat-184's committed pass is recorded as a measurement, with no band. This is not a weakening:
  exact equality on one host is a stricter test of "same text, opponent and judge" than any
  cross-host tolerance could be.
- Nothing else changes.

## Scoring log

### Scored 2026-09-24 23:28 IST --- P1 CONFIRMED, P2 TIE, P3 REFUTED / INCUMBENT WINS: the crossover is at `k=3`

All jobs exited `0` on host B (`output/logs/feat195_{G1ref,chat,J5,J6}.done`, launcher
`scripts/run_feat195.sh chat` with `GPU_CHAT=7`): the reference judged in `2` minutes, the three budgets
generated in `7`, each judge pass took under `3`. Scored from the per-prompt judge files by
`.venv/bin/python analysis/served_opponent.py --out results` (rows `G1-196`, `X-host` and the
`chat template, host B` rows of `results/served_opponent.csv`) and
`.venv/bin/python analysis/served_activity.py --out results`.

**Gates.** G0 PASS: `500` prompts in every new arm, every class at every budget `all_within_budget =
True` with `0` invariant violations (`output/feat196/chat_grid/h1_summary.csv`). G1 (restated before
launch) PASS: selection's per-prompt levels in J5 and J6 equal the host-B reference on `500/500` prompts.

**Cross-host measurement, no band.** The same judge on the same text agrees with feat-184's committed
local pass on `471/500` prompts for selection, `477` for its `n=1` control, `479` for the `k=1` meter and
`486` for the anchor; selection's level reads `0.367` on host B against `0.363` locally. G1 as first
written would therefore have failed on host arithmetic alone, which is why it was re-pointed before the
run.

| budget | `K/S_w` | active (forced) | reading | registered |
|---|---|---|---|---|
| `k=0.5` (J5 metered) | `0.63` | `62.5%` (`8.4%`) | `D3 = +0.1065 [+0.074, +0.1395]`, CONFIRMED | P1 CONFIRMED, **right** |
| `k=2` (J5 extra) | `2.50` | `2.5%` (`2.3%`) | `D5 = +0.030 [-0.009, +0.068]`, TIE | P2 TIE, **right** |
| `k=3` (J6 metered) | `3.75` | `1.1%` (`1.5%`) | `D3 = -0.051 [-0.0885, -0.0135]`, REFUTED | P3 meter ahead, **right** |
| `k=5` (J6 extra) | `6.26` | `0.23%` (`1.2%`) | `D5 = +0.073 [+0.034, +0.111]`, INCUMBENT WINS | P3 meter ahead, **right** |

On levels (selection minus meter, paired within each pass): `+0.1195 [+0.0925, +0.1465]` at `k=0.5`,
`-0.0170 [-0.046, +0.0125]` at `k=2`, `-0.0380 [-0.068, -0.008]` at `k=3`, `-0.0600 [-0.091, -0.029]` at
`k=5`. **The smallest budget at which the meter's interval lies above selection's is `k=3`, with
`K/S_w = 3.75`**: its certificate excludes nothing about a `50`-token window, as P4 recorded in advance.

**Manuscript, as registered.** Table 2's chat block reports all six budgets (`0.5`, `1`, `2`, `3`, `5`,
`10`); the four new rows are marked as judged on host B, with that host's selection and anchor levels in
the block header. Section 4 states that the meter overtakes selection from `k=3`, where `K/S_w = 3.75`.

**Correction, 2026-09-24 23:50 IST (descriptive P4 only; no registered reading changes).** The
"active (forced)" column above counted, on the chat arms, the one anchor-sampled step that follows the
risky model's `<|eot_id|>`, which is not a decode step (`results/served_activity_note.md`). Corrected by
`analysis/served_activity.py` stopping at the first end-of-text token: active `62.7%` (`8.1%` forced) at
`k=0.5`, `2.5%` (`1.8%`) at `k=2`, `1.1%` (`1.0%`) at `k=3`, `0.23%` (`0.80%`) at `k=5`. Table 2 prints
`62.7%` at `k=0.5`.

