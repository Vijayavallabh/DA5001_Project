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
