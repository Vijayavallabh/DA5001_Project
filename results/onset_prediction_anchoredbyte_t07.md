# Pre-registration: the authors' byte-level decoder at the authors' decoding settings (feat-205)

**feat-205.** Committed **2026-09-25**, before any token is decoded. Nothing above `## Scoring log` is
edited after the first byte is decoded.

## Why

Review 4 (sixth round, Q1) asks, at temperature `0.7` with repetition penalty `1.1`: "Does the meter gain
anything over its anchor at `k <= 0.5` (70B) and at `k = 0.1` per byte (AnchoredByte)?" feat-195
answered the token-level half. feat-187 ran AnchoredByte at temperature `1.0` with no penalty, where at
`k=0.1` it "is judged indistinguishable from Comma-7B alone". This arm runs it at the authors' settings.

## What runs

`analysis/anchoredbyte_decode.py` (the authors' package unmodified, `BytewiseAnchoredDecodingFactory`,
Comma-7B with the 70B base, `B_max = 800` bytes, batch `32`, seed `52`, the headline's `500` prompts, the
placement feat-187 recorded), with two new flags that pass straight to the authors' `generate()`:
`--temperature 0.7 --repetition-penalty 1.1`, which their factory applies to both log-probability
vectors before the fusion. Two budgets, each one run on three host-B cards:

- `k = 0.1` per byte (`K = 80` nats), the budget the review names;
- `k = 0`, the authors' own anchor-only path (their factory returns the safe distribution at
  `k_radius == 0`), which is the law the meter tilts away from **at these settings**. A token-level draw
  of Comma-7B stood in for it at temperature `1.0`; with a penalty that depends on the history, the
  authors' byte path is used instead so the control is theirs by construction.

-> `output/feat205/ab07/trajectories_k{0.1,0}_<class>.jsonl`.

**Judging**, one pass: `analysis/order_averaged_h2h.py --deecho --baseline-dir output/feat195/t07_8b
--sel-dir output/feat195/t07_pool64 --rewards results/selection_rewards64_t07.csv --metered-dir
output/feat205/ab07 --k 0.1 --anchor-dir output/feat205/ab07 --tag ab07_k01` (the anchor arm is read at
the filename token `0`): judge B, seed
`7717`, both orders, the temperature-`0.7` opponent and selection of feat-195 on the same host, the meter
at `k=0.1` with its own `k=0` as control.

## Gates

- **G0.** Both arms cover the `500` prompts; every record carries temperature `0.7` and penalty `1.1`.
- **G1.** No trajectory's realised spend exceeds `K + 1e-3`; the `k=0` arm spends `0`.
- **G2.** Selection's per-prompt levels equal feat-195 J1's
  (`results/order_averaged_h2h_per_prompt_t07_8b_k10.csv`: same texts, opponent, judge and host).

## Readings and predictions

- **A1.** The meter's gain over its own anchor (`D2`). *Predicted: DISSOLVES (interval straddles zero)*,
  as at temperature `1.0`.
- **A2, descriptive.** `D3` (TinyComma selection's gain minus the meter's, each over its own anchor);
  binding share and forced share of byte steps; `K/S_w` with `S_w` under the tempered Comma-7B
  (`analysis/regimes.py --temperature 0.7 --repetition-penalty 1.1`, run in feat-203's queue).

## What the manuscript does with each outcome, fixed now

Appendix `app:anchoredbyte` reports A1 at `0.7` beside the `1.0` reading. If A1 SURVIVES (the meter gains
over its anchor), the sentence that the meter at `k=0.1` is indistinguishable from Comma-7B alone is
scoped to temperature `1.0`, and Section 4's temperature paragraph says so.

## Excluded in advance

- Any other budget, temperature, penalty, seed or judge chosen after a byte is decoded.

## Scoring log
