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

### Addendum 2026-09-25 12:20 IST, before the control is re-run and before any judge call --- the `k = 0` control could not run as registered

The `k=0.1` arm ran (`08:59`-`12:11` IST, exit `0`). The `k = 0` control died at its first batch
(`output/logs/feat208_qab.log`): the authors' `generate()` routes every budget to `generate_byte`, which
asserts `k_radius not in {-1, 0}` with the message "Use generate_batched instead for your k-radius", and
the package contains no `generate_batched`. So the registered control, "the authors' own anchor-only path
at `k_radius == 0`", does not exist in the code we pinned; this is a defect in our specification, not a
reading (caution (w)), and it retires nothing.

**Replacement, fixed now.** The same path at `k = 1e-6` per byte (`K = 8 x 10^-4` nats): the budget
`(t+1)k` minus the prefix debt is negative at every byte, so the factory forces the safe distribution at
every step and the fused law is the anchor's, through the authors' own byte sampler at `0.7` and `1.1`.
**Gate G1', read before the judge:** every record has `steps_forced_safe == bytes_generated` (the anchor
served at every byte) and total spend `<= K + 1e-3`; if any record fails it, the control is not the
anchor and A1 is not read. The files (`trajectories_k1e-06_*`) are copied to
`output/feat205/ab07_anchor/trajectories_k0_*` so the judge's anchor loader reads them at its token `0`;
nothing else about the judge pass changes. Launcher `scripts/run_feat205_control.sh`, host B GPUs
`5,6,7` after feat-203's `70`B control releases them.

### Scored 2026-09-25 13:46 IST --- G1' PASS (the control is the anchor at every byte), A1 right (the meter's gain over its anchor DISSOLVES at `0.7` too)

All jobs exited `0` on host B (times IST): the `k=0.1` arm `08:59`-`12:11` (`scripts/run_feat208.sh ab`), the
`k = 1e-6` control `12:34`-`13:41` and the judge pass `13:41`-`13:43` (`scripts/run_feat205_control.sh`). The
control's three files were copied to `output/feat205/ab07_anchor/trajectories_k0_*` byte-identically (`cmp`).
Scored by `.venv/bin/python analysis/score_feat203_206.py --only 205` -> `results/anchoredbyte_t07.csv`.

**Gates.** G0 PASS: both arms cover the `500` prompts and every record carries temperature `0.7` and penalty `1.1`.
G1 PASS: the largest `k=0.1` spend is `74.90` against `K = 80`; the control spends `0.0` against `8 x 10^-4`.
**G1' PASS**: in every record of the control the forced steps equal the generated bytes, so A1 may be read. G2
PASS: selection's per-prompt levels equal feat-195 J1's on `500` of `500`.

| reading | value | registered | verdict |
|---|---|---|---|
| A1 the meter's gain over its own anchor at `k=0.1` (`D2`) | `+0.008 [-0.008, +0.0245]`, DISSOLVES | DISSOLVES | **right** |

**Descriptive (A2), no band.** `D3`, TinyComma selection's gain over its `n=1` control minus the meter's over its
anchor, all at `0.7`/`1.1`: `+0.099 [+0.068, +0.129]`, REVERSAL CONFIRMED. (At `1.0` the committed `D3` set the
meter against **Comma-7B's** selection, `+0.1385`; the registered pass here uses feat-195's temperature-`0.7`
TinyComma selection, so the two `D3`s are not the same comparison and are not set side by side.) `S_w` under the
tempered Comma-7B `155.34` nats, against `140.71` at `1.0` by the same construction on the same `758` works, so
`K/S_w = 0.515` at `k=0.1` (`0.569`). The registered per-record mean binding share is `0.3302`.

**Post hoc, descriptive** (rows appended after the registered ones, which were checked unchanged). Pooled over
bytes, the construction `results/anchoredbyte.csv` uses for the `1.0` figures the paper quotes, the meter binds on
`22.9%` of byte steps and is forced to the anchor on `14.1%` (`18.4%` and `13.4%` at `1.0`); the control is forced on
`100%`. Both arms return no byte on the same `66` prompts (`33` at `1.0`): the prefix debt (`5.05`-nat minimum)
forces the anchor for the opening bytes at `k=0.1`, so both arms share their first `40` characters on all `433`
prompts with text, and on `140` prompts the meter serves exactly the control's text. **Five of those `140` identical
pairs are judged differently** (`0.25` against `0.5`): each arm is judged in its own batches, so batch composition
alone moves a greedy bf16 judge on identical input, the mechanism feat-209's post-hoc shift names.

**Manuscript, as registered.** A1 DISSOLVES, so `app:anchoredbyte` reports it at `0.7` beside the `1.0` reading and
nothing is scoped; Section 4 is unchanged.
