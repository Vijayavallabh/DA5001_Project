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
