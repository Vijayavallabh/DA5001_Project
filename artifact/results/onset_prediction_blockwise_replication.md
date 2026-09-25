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

### Scored 2026-09-25 12:24 IST --- P1 right (the `L=10` advantage holds on a fresh draw), P2 WRONG in selection's favour (`L=25` wins too), P3 right; judge G agrees on both draws

All jobs exited `0` on host B (`scripts/run_feat208.sh`): judge G post hoc on feat-201's texts `09:56`-`10:24`
IST; the re-draws `blk200n64s2` `10:18`-`10:30`, `blk25n64s2` `10:08`-`10:50`, `blk10n64s2` `10:24`-`12:13`; each
judged by B and then G. Scored by `.venv/bin/python analysis/blockwise_replication.py --out results` ->
`results/blockwise_replication.csv`.

**Gates.** G0 PASS (`500` records and complete block logs for the three arms; three generation logs print
`temperature=1.0 top_k=0 top_p=1.0`). G1 PASS: on every prompt where both draws serve text, the new draw
serves a different text from feat-201's (`345`, `338` and `333` prompts at `L = 10, 25, 200`). G2 PASS:
meter and anchor levels equal feat-198's on `500/500` in all three judge-B passes.

| reading | value | registered | verdict |
|---|---|---|---|
| P1 judge B, new draw, `L=10` minus once | `+0.0435 [+0.020, +0.067]`, INSTALLMENTS WIN | INSTALLMENTS WIN | **right** |
| P2 judge B, new draw, `L=25` minus once | `+0.033 [+0.011, +0.0555]`, INSTALLMENTS WIN | TIE | **wrong** (in selection's favour) |
| P3 judge G, new draw, `L=10` minus once | `+0.0635 [+0.033, +0.0945]` | positive | **right** |

**Descriptive, no band.** Judge G, new draw, `L=25` minus once: `+0.0595 [+0.028, +0.0905]`. Gains over the
anchor on the new draw: judge B `+0.161`, `+0.1505`, `+0.1175` at `L = 10, 25, 200`; judge G `+0.2735`,
`+0.2695`, `+0.21`. **Post hoc**, judge G on feat-201's own texts: `+0.083 [+0.0525, +0.114]`,
`+0.0635 [+0.0325, +0.0955]` and `+0.031 [+0.001, +0.062]` at `L = 10, 25, 50` minus once. Served medians
`89`, `77` and `59` words with `98`, `114` and `104` empty answers.

Both registered draws and both judges put installments of `10` and `25` tokens above one choice among `64`
whole drafts; no reading is pooled.

**Manuscript, as registered.** Section 3 says the `L=10` advantage held on a fresh draw and quotes both
readings; the conclusion keeps its installments clause; the appendix reports judge G beside judge B.
