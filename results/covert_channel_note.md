# Note: the colluding-scorer channel that the `log n` certificate prices (feat-192; descriptive)

Simulation over committed candidate pools; no generation, no judge. Committed **2026-09-24, before
the script was run.** Nothing above `## Scoring log` is edited afterwards. The channel's rate for
pseudorandom hashes is a closed form, so the only empirical inputs are the pools' duplicate structure
and the protected windows' lengths; the expected reading is stated so the result can be checked.

Produced by `.venv/bin/python analysis/covert_channel.py --out results` -> `results/covert_channel.csv`
(per `n`) and `results/covert_channel_windows.csv` (per `n` and encoding).

## Why

Referees asked for a multi-query reconstruction attack with a colluding scorer transmitting up to
`log2 n` bits per response. Proposition 1 bounds the served law by `q(y) <= n p_s(y)` for **any**
score, so one response carries at most `log2 n` bits about anything, and `m` responses at most
`m log2 n` --- the composition remark's union factor. That is an upper bound; this shows the channel
is real, measures how much of the bound a simple scheme realises, and counts the responses it needs
to move one protected `50`-token window.

**What it is not.** No served response is a copy of anything: each is one of the anchor's own draws,
verbatim. The message moves through *which* draw is served, and the receiver needs the key.

## Construction (fixed now)

- **Pools.** `output/phase5/sel_anchor64` (TinyComma, `500` prompts x `64` draws), texts recovered
  with `served_generation`. Each response samples a prompt and a fresh `n`-subset of its `64` draws;
  identical draws hash identically, which lowers the effective `n`, and the distinct-draw fraction is
  reported.
- **Scorer.** HMAC-SHA256 of (response index, draw) with a shared key, read as `u in [0,1)`; an
  erasure cell `[0, eps)` and `2^b` equal cells on `[eps, 1)`; serve a draw in the wanted cell, else an
  erasure draw, else the lowest cell (an undetected error). `(b, eps)` maximise `b P(match)` subject to
  `P(undetected error) <= 1e-3` **from the i.i.d. formulas only**. `n in {2, 4, 8, 16, 32, 64}`,
  `20,000` simulated responses each.
- **Messages.** The `100` protected passages of the extraction table (`attack_train`, `20`-token
  seed, `selection_extraction.build`), the first `50` tokens of each target (TinyComma's Llama-3
  tokenizer), under two public codes with a `16`-bit length header: raw UTF-8 and raw `lzma`.
  Transmitted **end to end**, decoded, and compared byte for byte.
- **Beside it.** `m log2 n` bits after `m` responses; the responses a capacity-achieving code would
  need, `bits / log2 n`; and the `400`-nat per-user odometer's cap, `floor(400 / ln n)` responses and
  `400 / ln 2 = 577` bits whatever `n`.

## Expected reading

Realised bits per response within `5%` of the analytic rate at every `n`, below `log2 n` everywhere
(a rate above it would be a bug), efficiency well below `1` and rising slowly with `n`. Every window
decoded exactly. At `n=64` a window costs on the order of a hundred responses under `lzma` --- inside
the odometer's cap at `n=64` (`96` responses), which is the point: the odometer, not the per-response
certificate, is what bounds a patient user.

## Excluded in advance

- Tuning `(b, eps)` on the simulated outcome; calling any served response a copy of a passage.

## Scoring log
