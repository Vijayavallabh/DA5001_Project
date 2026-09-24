# Pre-registration: where to spend a fixed pathwise certificate --- on the draw, on the token, or split (feat-189)

**feat-189.** Committed **2026-09-24**, before any draft of this arm was generated. Nothing above
`## Scoring log` is edited afterwards.

## Why

Review 4 (B.1) asks for the hybrid: select among drafts from a pathwise-budgeted meter. If every draft
satisfies `q_B(y) <= e^B p_s(y)` for every `y` (the pathwise decoder's guarantee, `R/K <= 1` on all
`16,200` pathwise trajectories on record) then serving the best of `n` such drafts satisfies
`q(y) <= n e^B p_s(y)`, a pathwise certificate of `log n + B` --- Proposition 1 applied to a pool drawn
from `q_B` instead of `p_s`. So one budget `C` can be split between the draw (`log n`) and the token
(`B = C - log n`). Pure selection puts all of it on the draw; the metered decoder all of it on the
token. This measures three splits of **`C = log 64 = 4.159` nats**, the headline's own certificate.

## What runs (local GPU 0; `scripts/run_hybrid.sh`)

`h1.py --constraint pathwise --no-prefix-debt`, the headline's `500` prompts (caps `200/150/150`),
temperature `1`, `T_max = 200`, batch `64`, default seeds, the per-token `k = B / 200` so that
`K = kT = B`:

| arm | `n` | `B` | `k` (CLI token) | trajectories |
|---|---|---|---|---|
| `hyb8` | `8` | `log 8 = 2.0794` | `0.010397` | `4,000` |
| `hyb2` | `2` | `log 32 = 3.4657` | `0.017329` | `1,000` |
| `pw1` | `1` | `log 64 = 4.1589` | `0.020794` | `500` |

and, from the committed pool, `sel64` (`n=64`, `B=0`), `sel8` and `sel1`. Prefix debt is off because the
certificate here is on the continuation; with it on, `B < delta_init ~ 6` nats would leave the budget
negative and every draft would be the anchor. Selection among drafts uses the committed reward
(Qwen2.5-7B-Instruct, `selection_scaling.py --rewards-only --k-token <k>`, batch `8`).

Judge: `analysis/levels_pass.py`, judge B, the committed opponent, both presentation orders, de-echoed,
one pass (`--tag hybrid`); contrasts by paired bootstrap (`--score hybrid`).

## Gates

- **G0.** Every draft's realised pathwise spend is `<= B + 1e-3` (the per-trajectory rule of the
  Working Rules); a violation voids the arm.
- **G1.** `sel64`, `sel8`, `sel1` reproduce feat-186's per-prompt levels exactly (`levels_pass.py` is
  deterministic per arm; a mismatch means the pass is not the committed construction).

## Bands (readings as `levels_pass.py --score` writes them: ABOVE / BELOW / STRADDLES ZERO)

- **H1, `hyb8 - sel64`, the equal-certificate split.** Predict **BELOW ZERO**: two nats spread over
  `200` tokens tilt a draft very little, and going from `8` draws to `64` is worth `+0.088` on this
  anchor (feat-131).
- **H2, `pw1 - sel64`, all of it on the token.** Predict **BELOW ZERO**: `4.16` nats is the metered
  decoder below its every published budget.
- **H3, `hyb2 - sel64`.** Predict **BELOW ZERO**.
- **H4, `hyb8 - sel8`, the tilt at fixed `n`.** Predict **STRADDLES ZERO**.

## What the manuscript does with each outcome, fixed now

One paragraph in the appendix beside Proposition 1's composition remark, and one clause in Section 3:
at a fixed certificate, the reading of H1--H3 says whether the budget is better spent on the draw. If
any of H1--H3 reads ABOVE ZERO, the paper says a split beats pure selection at that certificate, and
the abstract's framing (spend it on the draw) is qualified in the same sentence.

## Excluded in advance

- Any other split, budget, `n`, schedule or judge chosen after a judge call.
- Turning prefix debt on for one arm and not another.

## Scoring log

### Scored 2026-09-24 14:50 IST --- H1, H2, H3 BELOW ZERO and H4 STRADDLES, as predicted: at a fixed certificate, spend it on the draw

Generation `output/hybrid/pw_n{8,2,1}` (local GPU 0, `scripts/run_hybrid.sh`), rewards
`results/selection_rewards{8,2}_hybrid.csv`, judge `analysis/levels_pass.py --tag hybrid`, contrasts
`--score hybrid` -> `results/levels_hybrid{,_per_prompt,_contrasts}.csv`.

**G0 PASS**, on the quantity the pathwise decoder bounds: the realised log-ratio of every served path
is within `B` (max `1.9459` against `log 8 = 2.0794`, `3.2658` against `log 32`, `3.6105` against
`log 64`), and the harness's own invariant holds on all `5,500` drafts. The field `total_spend`
exceeds `B` on `79` of the `n=8` drafts; it is the KL charge, which pathwise accounting does not cap,
so it is not the gate's quantity and the registration's wording ("realised pathwise spend") meant the
ratio. The drafts barely leave the anchor: median KL `0.125` nats at `B = log 8`.
**G1 PASS**: `sel64`, `sel8`, `sel1` equal feat-186's per-prompt levels on `500` of `500` prompts.

| arm | `n` | `B` | certificate | level |
|---|---|---|---|---|
| `sel64` | `64` | `0` | `log 64` | `0.5550 [0.5340, 0.5760]` |
| `sel8` | `8` | `0` | `log 8` | `0.5115` |
| `sel1` | `1` | `0` | `0` | `0.4535` |
| `hyb8` | `8` | `log 8` | `log 64` | `0.4995 [0.4780, 0.5210]` |
| `hyb2` | `2` | `log 32` | `log 64` | `0.4610` |
| `pw1` | `1` | `log 64` | `log 64` | `0.4400 [0.4190, 0.4610]` |

| band | contrast | reading | predicted |
|---|---|---|---|
| H1 | `hyb8 - sel64` | `-0.0555 [-0.0780, -0.0330]` BELOW ZERO | BELOW ZERO, right |
| H2 | `pw1 - sel64` | `-0.1150 [-0.1380, -0.0915]` BELOW ZERO | BELOW ZERO, right |
| H3 | `hyb2 - sel64` | `-0.0940 [-0.1170, -0.0715]` BELOW ZERO | BELOW ZERO, right |
| H4 | `hyb8 - sel8` | `-0.0120 [-0.0300, +0.0060]` STRADDLES | STRADDLES, right |

**The ordering is monotone in how much of the certificate goes to the draw**: `pw1` (`n=1`) `0.4400`,
`hyb2` `0.4610`, `hyb8` `0.4995`, `sel64` (`n=64`) `0.5550`. And the pathwise meter given the whole
certificate, `4.16` nats, is judged **no better than a single anchor draw** (`0.4400` against
`0.4535`).

**One confound, disclosed and not scored:** the drafts are the first correctly recorded pool, and the
committed reward rates an empty completion above a typical one (median `-12.1` against `-26.8`), so
`hyb8` serves an empty text on `39.0%` of prompts against `sel8`'s `21.8%`. H4 straddles zero despite
it, so the tilt is at best worth what the extra empties cost; the registered readings stand as scored.
