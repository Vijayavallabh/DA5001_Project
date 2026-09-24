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

### Scored 2026-09-24 16:15 IST --- B1, B2, B3 CONFIRMED; G0, G1, G2 pass

Generation on host B (`k=0.5` then `2` on GPUs 0--2, `k=0.1` on 3,5,6; batch `32`, no OOM retry
needed); judge passes on host B GPU 7 (`k=0.5` re-run on the same card after it OOM'd beside
FActScore, so all three passes share silicon); `analysis/anchoredbyte_score.py --out results` ->
`results/anchoredbyte.csv`.

**G0 PASS** (every arm covers the `500` prompts; `full_text` is prompt plus generation on every record),
**G1 PASS** (largest realised spend `74.84` of `K = 80`, `362.44` of `400`, `466.34` of `1600`),
**G2 PASS** (selection's per-prompt levels identical across the three passes).

| `k` | `K` | `K/S_w` | binds | forced | spend median | meter `D2` | `D3` selection `-` meter | band |
|---|---|---|---|---|---|---|---|---|
| `0.1` | `80` | `0.57` | `18.4%` | `13.4%` | `47.2` | `+0.0200 [-0.0025, +0.0420]` | `+0.1385 [+0.1155, +0.1620]` | B1 **CONFIRMED** |
| `0.5` | `400` | `2.86` | `2.8%` | `2.6%` | `78.9` | `+0.0345 [+0.0120, +0.0575]` | `+0.1240 [+0.1010, +0.1470]` | B2 **CONFIRMED** |
| `2` | `1600` | `11.45` | `0.6%` | `0.6%` | `85.5` | `+0.0240 [+0.0010, +0.0475]` | `+0.1345 [+0.1105, +0.1590]` | B3 **CONFIRMED** |

`S_w = 139.68` nats (the audited anchor's `159.83` rescaled by the two anchors' per-character
surprisal). Selection at Comma-7B, `n=64`: `D1 +0.1585 [+0.1355, +0.1825]`, level `0.6245` against
Comma-7B alone `0.4660`. **B4, descriptive:** the unconstrained `70`B base gains `-0.0110 [-0.0340,
+0.0120]` over Comma-7B alone, and selection beats it by `0.1695 [0.1445, 0.1945]`; `33` of the `500`
AnchoredByte generations are empty at every `k`. Every prediction was right. **What the manuscript
does, as fixed above:** the rows go to the appendix twin of Table 1 (`tab:anchoredbyte`; the body has
no room), and since every band reads CONFIRMED the abstract's "the authors' own `70`B pair included"
gains their byte-level decoder at its recommended pair.
