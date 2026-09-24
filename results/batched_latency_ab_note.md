# Note: one request at the authors' recommended pair --- Comma-7B selection against AnchoredByte (descriptive)

Committed **2026-09-24 16:20 IST**, before any cell below is timed. Nothing above `## Scoring log` is
edited afterwards. Descriptive: no band is registered, because feat-190's bands are about the token-level
pairs and this cell was added after they were scored; the expectation is stated so it can be wrong.

## Why

Review 4 (for a stronger paper): "show the reversal at matched wall-clock on at least one 7B+ licensed
pair". feat-187 judged the reversal at Comma-7B against the authors' own byte-level decoder
(`results/anchoredbyte.csv`: selection wins by `+0.124` to `+0.139`); feat-190 timed only the
token-level pairs. This times the same two mechanisms at that pair, with feat-190's protocol.

## Cells (host B, `analysis/batched_latency.py`, appended to feat-190's log as `part=ab`)

- **METAB**: `BytewiseAnchoredDecodingFactory` (the authors' package at `a12ecd9`), Comma-7B with the
  `70`B base, `k = 0.5` per byte (their recommended point), `max_new_tokens = 200` (so `B_max = 800`
  bytes), loaded exactly as `analysis/anchoredbyte_decode.py` loads it, on GPUs 0, 1, 2, `W in {1, 8}`.
- **SEL**: Comma-7B, one `generate(num_return_sequences=n)` for `n in {1, 8, 64}` and one reward pass
  with the committed scorer, on GPU 0 (the meter's first card) right after the meter, `W in {1, 8}`.

Everything else as feat-190: the headline's neutral prompts, rep `r` on prompts `[rW, (r+1)W)`, `3`
reps, one untimed warm-up, models loaded once and the load timed apart. **Unlike every feat-190 cell,
AnchoredByte stops at end-of-text**, so each cell also records the bytes it served, and the comparison
is read both per request and per served byte. A cell that does not fit its card (selection at `n=64`,
`W=8` holds `512` Comma-7B sequences beside two 7B models) is reported as not measured, never re-run on
a different layout.

## Expectation (not a band)

`SEL(n=64) / METAB` below `1` at `W = 1`, per request and per served byte: the byte-level loop forwards
a 7B and a 70B for every token of either tokenizer, selection one 7B in one batched call.

## Scoring log

### Read 2026-09-24 16:30 IST --- selection at Comma-7B serves a request in `0.061x` AnchoredByte's time

Host B, 12:41--12:52 CEST, GPUs 0, 1, 2 (meter) and 0 (selection, right after); `analysis/batched_latency.py
--report` -> `results/batched_latency.csv` (part `ab`) and the `AB` rows of
`results/batched_latency_bands.csv`.

| per-request seconds (bytes served) | `W = 1` | `W = 8` |
|---|---|---|
| AnchoredByte, Comma-7B + 70B, `k = 0.5` | `68.69` (`800`) | `10.91` (`713`) |
| selection at Comma-7B, `n = 1` | `3.09` (`812`) | `0.423` (`825`) |
| selection at Comma-7B, `n = 8` | `3.12` (`822`) | `0.525` (`789`) |
| selection at Comma-7B, `n = 64` | `4.19` (`794`) | not measured: out of memory |

`SEL(64) / METAB = 0.061` at one request per call (`0.061` per served byte too; the byte-level meter
served its full `800`-byte budget on all three repeats), and `SEL(8) / METAB = 0.048` at eight per call.
The expectation held. Two things read against the cell: the `n = 64` selection cell has a `21%` repeat
spread (its first repeat took `4.19` s of generation against `3.36` and `3.34`), which is far from
mattering at a ratio of `0.06`; and at eight requests per call `n = 64` did not fit one 80 GB card beside
the reward model (`512` Comma-7B sequences), so that cell is reported as not measured, as fixed above.
