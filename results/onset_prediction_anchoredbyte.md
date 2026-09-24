# Pre-registration: the authors' own AnchoredByte at their recommended pair, Comma-7B + Llama-3.1-70B base (feat-187)

**feat-187.** Committed **2026-09-24**, before any AnchoredByte trajectory beyond a 9-prompt smoke
was generated and before any judge call. Nothing above `## Scoring log` is edited afterwards.

## Why

Review 4 (marked most important): the mechanism is not tested as published. He et al. meter every
anchor but TinyComma with **AnchoredByte**, byte-level fusion through ByteSampler, and the pair their
README recommends is Comma v0.1-2t (Comma-7B) with Llama-3.1-70B at `k = 0.5`. Our strongest selection
results are at Comma-7B (feat-095: `+0.111 [+0.072, +0.148]` at `n=64`), and that anchor has never had
a metered counterpart, because a Comma tokenizer cannot be token-fused with a Llama-3 risky model.

## What runs

**Generation, host B, GPUs 0--2.** The authors' own implementation, unmodified: package
`anchoreddecode` from `github.com/jacqueline-he/anchored-decoding` at commit `a12ecd9` (2026-05-06;
cloned into `ext/`, gitignored, installed editable on host B, whose copy of `src/` hashes identically), `BytewiseAnchoredDecodingFactory`, called through `analysis/anchoredbyte_decode.py`, which
chooses only the prompts, the serving configuration and the output layout. Safe `common-pile/
comma-v0.1-2t`, risky `unsloth/Meta-Llama-3.1-70B` (the base checkpoint every 70B arm on record uses),
bf16. The headline's `500` prompts (`output/sweep_plain`'s `k=-1` prompt ids), prompt text from
`load_prompt_corpus` with the `Complete the prefix:` header, **temperature `1.0`, no repetition
penalty, no chat template** --- Table 1's serving configuration, not the authors' `0.7`/`1.1`. The
authors' defaults otherwise: prefix debt on, `n = 5`, scaled by `TOKEN_TO_BYTE = 4`;
`max_new_tokens = 200`, which their factory turns into `B_max = 800` bytes, so the certificate is
`K = 800 k` nats. Per-byte budgets `k in {0.1, 0.5, 2}` from their grid
`{0.1, 0.5, 1, 1.5, 2, 3, ...}`: `K = 80, 400, 1600`. Batch `32` (`16` if it does not fit), seed
`52 * 100000 + batch index`.

One placement change, forced by hardware and not touching arithmetic: the sampler keeps its budget on
the safe model's device, prefix debt on the risky model's embedding device, and fuses on the risky
logits' device, so all three must be one card. The authors get that by loading both models on one
141 GB H200; on 80 GB H100s the risky model's embedding, final norm and head sit on card 0 beside
Comma-7B and only its 80 decoder layers are spread over cards 1 and 2. A 9-prompt smoke at `k=0.5`
ran end to end (`output/anchoredbyte/smoke70`, `10.1` s/prompt at batch 9) and is not part of any
reading.

**Judge, one pass per `k`,** `analysis/order_averaged_h2h.py --deecho`, judge B (Phi-3.5-mini),
seed `7717`, both presentation orders, the **committed opponent** (`output/sweep_plain` `k=-1`, lowest
seed), exactly the construction of feat-184/185:

- selection: Comma-7B best-of-`64` (`output/phase5/sel_comma7b_64`, cached picks from
  `results/selection_rewards64_comma7b.csv`) and its control `n=1`, the pool's rank-0 draw;
- metered: AnchoredByte at `k`; its control Comma-7B alone, which is the same rank-0 draw (ByteSampler
  samples the anchor's own distribution exactly, so a token-level draw of Comma-7B is a draw of the
  byte-level anchor);
- extra arm: the unconstrained 70B base (`output/phase5/imit_llama70b` `k=-1`), what the meter tends
  to as `k` grows.

Tags `ab70_k0.1`, `ab70_k0.5`, `ab70_k2`.

## Gates

- **G0.** Every arm covers the `500` prompts; AnchoredByte's `full_text` is prompt plus generation on
  `100%` of records (the authors' factory returns exactly that).
- **G1.** No trajectory's realised spend exceeds `K + 1e-3` (the authors' loop asserts the per-step
  bound; this reads the total).
- **G2.** Selection's per-prompt levels are identical across the three passes (same text, same
  opponent, greedy judge).

## Bands (`D3` = selection's gain minus the meter's, paired; `D1`, `D2` each arm's own gain)

Readings as the script writes them: **CONFIRMED** (`D3` interval above zero), **REFUTED** (below),
**UNRESOLVED**.

- **B1, `k = 0.1` (`K = 80`, below a 50-token window's surprisal, the only budget here that
  certifies anything about one).** Predict **CONFIRMED**: at the token level the meter at its one
  informative budget was inseparable from its anchor at both pairs (feat-184/185).
- **B2, `k = 0.5` (`K = 400`), the authors' recommended operating point.** Predict **CONFIRMED**:
  at temperature `1.0` the 70B base was judged no better than a TinyComma draw (feat-185 C4,
  `+0.0190 [-0.004, +0.042]`), and Comma-7B is the stronger anchor.
- **B3, `k = 2` (`K = 1600`).** Predict **CONFIRMED**, for B2's reason.
- **B4, descriptive.** Each `k`'s binding share (byte steps with risky weight `< 1`) and forced share,
  the realised spend against `K`, and `D4`, the unconstrained 70B's gain over Comma-7B alone.

## What the manuscript does with each outcome, fixed now

A new block of Table 1 (or its appendix twin if the body has no room): "AnchoredByte, Comma-7B +
70B base", the three `k` with `K`, selection's `D3` beside each. If every band reads CONFIRMED, the
abstract's "the authors' own `70`B pair included" gains "and their byte-level decoder at their
recommended pair". Any REFUTED is reported as measured, in the same sentence, and the judged claim is
scoped to it.

## Excluded in advance

- Another judge, opponent, seed, budget or temperature chosen after a judge call.
- Pooling with the token-level passes, or setting these levels against another pass's.
- Reading `k = 0.1` as a statement about other window lengths.

## Scoring log
