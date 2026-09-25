# Pre-registration: selection in installments, a fresh draw and a second judge (feat-208)

**feat-208.** Committed **2026-09-25 09:55 IST**, after feat-201 was scored and before any token of this
arm is drawn. Nothing above `## Scoring log` is edited after the first registered draw.

## Why

feat-201 found installments beating one choice at `64` draws per block (`L=10`: `+0.0415 [+0.0185,
+0.0645]`; `L=25`: `+0.0245 [+0.001, +0.0475]`), against its own registered prediction of a tie. Both are one
draw under one judge, and `L=10` sits about `1.8` interval half-widths from zero, the range where a paired
difference on record has failed to reproduce (caution (ap): `1.7` did not, `2.1` and `2.43` did). Before the
manuscript leans on it, the same comparison is drawn again and read by a second judge.

## What runs

- **A disjoint draw.** `analysis/blockwise_selection.py` exactly as feat-201 ran `blk10n64`, `blk25n64` and
  `blk200n64` (value scorer, `n=64`, generation batch `256`, scoring batch `32`), with `--seed 20260926` in
  place of `20260925`, tags `blk10n64s2`, `blk25n64s2`, `blk200n64s2`, -> `output/feat208/`.
- **Judge B** on each, as feat-201 judged (`order_averaged_h2h.py --deecho --extra-dir ... --tag
  blockwise_<arm>`), host B.
- **Judge G** (`google/gemma-2-27b-it`, `--device-map auto`, the committed judge-G pass's command, the most
  order-consistent judge on record) on the three new arms, tags `blockwise_<arm>_judgeG`.
- **Post hoc, descriptive, no band:** judge G on feat-201's own `blk10n64`, `blk25n64`, `blk50n64` and
  `blk200n64`, run before this draw starts. It is reported as post hoc because feat-201's excluded
  outcomes named "any other judge chosen after a draw is read".

## Gates

- **G0.** Each new arm has `500` records and its block log is complete; every generation log prints the
  pure-sampling configuration.
- **G1.** The two draws differ: the new arm's served text differs from feat-201's on at least `90%` of the
  prompts on which both are non-empty.
- **G2.** In each judge-B pass the meter's and the anchor's per-prompt levels equal feat-198's (same host,
  same judge, same texts), as in feat-201.

## Readings and predictions (paired over prompts, `10,000` resamples, `95%`)

- **P1, judge B, new draw.** `u(blk10n64s2) - u(blk200n64s2)`: INSTALLMENTS WIN, TIE or ONCE WINS.
  *Predicted: INSTALLMENTS WIN.*
- **P2, judge B, new draw.** `u(blk25n64s2) - u(blk200n64s2)`. *Predicted: TIE* (feat-201's lower end was
  `+0.001`).
- **P3, judge G, new draw.** The `L=10` difference has a positive point estimate. *Predicted: positive.*
- **Descriptive.** The same contrasts under judge G for the new draw at `L=25`; each arm's gain over the
  anchor; empty answers and median words.

## What the manuscript does with each outcome, fixed now

- P1 INSTALLMENTS WIN: Section 3 says the installment advantage at `L=10` held on a fresh draw, and quotes
  both readings; the conclusion keeps the installments clause.
- P1 TIE or ONCE WINS: Section 3 quotes feat-201's reading beside the fresh draw's and says the advantage
  is not established; the conclusion says installments are certified at a few multiples of `log n` per
  window and "may" buy more, nothing stronger.
- Judge G is reported beside judge B in the appendix whatever it reads. No reading is pooled with
  feat-201's.

## Excluded in advance

- Pooling the two draws; a third draw chosen after this one is read; any other `L`, `n`, scorer or judge.

## Scoring log
