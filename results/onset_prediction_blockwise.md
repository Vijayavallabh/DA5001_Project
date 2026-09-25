# Pre-registration: selection spent in installments (feat-201)

**feat-201.** Committed **2026-09-25**, after a six-prompt launch check of the code path and before any
registered arm draws a token. Nothing above `## Scoring log` is edited after the first registered draw.

## Why

Review 4 (sixth round, Q3 and W3(i)) asks how selection spent in installments, `n` anchor draws per
block of `L` tokens with one served, trades utility against its window certificate
`(ceil(w/L)+1) log n`, and whether it dominates selection spent once. Review 3 calls it the most
important missing cell. Section 3 states the certificate and Limitations says we did not measure it.
This arm measures it, and adds the two variants the same reports raise: the risky model as the
scorer ("let the risky model act only through the certified channel", Review 4, Oral route) and the
adversarial memoriser as the scorer (does leakage stay at zero when the adversary chooses `20` times
rather than once).

## What runs

`analysis/blockwise_selection.py`, host B, one arm per invocation. Anchor `TinyComma-1.8B` **alone**
(no risky model in the draw path), pure sampling (temperature `1.0`, `top_k=0`, `top_p=1.0`, set
explicitly because the checkpoint's `generation_config.json` asks for `0.6`/`0.9`; each log prints the
configuration `generate()` resolves), `T_max = 200`, seed `20260925`, generation batch `256`, scoring
batch `32`. Candidates are token ids cut after the first end-of-text; the served text is the chosen
ids decoded. Prompts, their `prefix_text` and their token counts are the committed headline pool's
(`output/phase5/sel_anchor64`, `500` prompts: `200` neutral, `150` factual, `150` creative), and the
script asserts every token count.

**The scorer (primary, `--scorer value`).** The committed pointwise reward
(`analysis/selection_scaling.score_rewards`: Qwen2.5-7B-Instruct, `log p(Yes) - log p(No)` on the
committed template) of each candidate **completed to `T_max` by one anchor rollout**, with the corpus
prompt (harness header removed) in the Instruction slot. The rollout is an input to the score only;
the served block is one of the `n` draws, so Proposition 1 holds per block. At `L = T_max` there is
nothing to roll out and the rule is whole-output selection through the same code.

The launch check (six prompts, `L=50`, `n=8`) showed why the rollout is needed: scored on the prefix
alone, a candidate that ends inside the first block is a complete answer competing against truncated
ones, and four of six prompts stopped after one block. With the rollout five of six still did, so the
committed reward prefers short complete answers whichever way it is asked; that is a property of the
scorer the whole-output arm shares, and it is reported, not corrected.

| arm | `L` | `n` per block | blocks | certificate, whole output (nats) | certificate, any 50-token window (nats) | draw tokens per prompt, at most |
|---|---|---|---|---|---|---|
| `blk10n64` | 10 | 64 | 20 | `83.18` | `24.95` | `12,800` |
| `blk25n64` | 25 | 64 | 8 | `33.27` | `12.48` | `12,800` |
| `blk50n64` | 50 | 64 | 4 | `16.64` | `8.32` | `12,800` |
| `blk200n64` (once) | 200 | 64 | 1 | `4.16` | `4.16` | `12,800` |
| `blk100n8` | 100 | 8 | 2 | `4.16` | `4.16` | `1,600` |
| `blk67n4` | 67 | 4 | 3 | `4.16` | `2.77` | `800` |
| `blk34n2` | 34 | 2 | 6 | `4.16` | `2.08` | `400` |

Every window certificate is far below the protected window's anchor surprisal, `S_w = 159.8` nats.
The first four arms hold the draws fixed and vary `L` (Review 4's `L in {10, 25, 50, T}`); the last three
hold the **whole-output certificate** at `log 64` and spend it in `2`, `3` or `6` installments
(`B log K = log 64`).

**Further arms.**
- `blk50n64_reward`: `L=50`, `n=64`, scored on the prefix alone (`--scorer reward`), the naive rule, as
  an ablation of the rollout.
- `blk10n64_planner`: `L=10`, `n=64`, scored by the risky model (`Llama-3.1-8B-Instruct`, plain text as in
  the headline) as planner: `log p_r(block | context) - log p_s(block | context)` on the shared Llama-3
  token ids (the script asserts the two vocabularies agree), argmax served.
- `blk10n64_memoriser`: `L=10`, `n=64`, the adversarial rule of `analysis/selection_extraction.py` (per-token
  mean log-probability of the block under the LoRA memoriser `output/memorizing_llama8b`) on the `100`
  `attack_train` passages that memoriser was fine-tuned on, `20`-token seeds with the header it was
  trained with; near-verbatim recall and ROUGE-L against the passage's continuation ->
  `results/blockwise_extraction_blk10n64_memoriser.csv`. Its `k=-1` and `k=0` baselines on the same
  passages and seeds are the committed `results/selection_extraction_per_passage.csv`
  (`risky_alone_recall`, `recall_n1`); the scorer asserts the passage sets agree.
- `blk200n1`: the anchor alone through this code path (gate G1, and judged as a null arm).
- `blk200n1_shipped`: the same on the first `100` prompts under the checkpoint's shipped sampling
  (`--shipped-sampling`, temperature `0.6`, `top_p 0.9`): the deliberate defect G1 must detect.

**Judging.** Every pool arm through `analysis/order_averaged_h2h.py --deecho --extra-dir
output/feat201/<arm> --extra-token <arm> --extra-name <arm> --tag blockwise_<arm>`: judge B
(Phi-3.5-mini), seed `7717`, both orders, the headline's opponent (the risky model's lowest-seed draw)
and its anchor control. An order-averaged level is a deterministic function of the prompt, the two
texts and the judge (caution (ap)), so levels from different passes on one host are compared per prompt.

## Gates

- **G0.** Every pool arm has one record for each of the `500` prompts, none longer than `200` tokens, and a
  block log with one row per active prompt per block; the memoriser arm covers the `100` passages.
- **G1, the sampling law.** Mean anchor surprisal of the served tokens (nats per token, the text
  re-tokenised with its prompt as context, `analysis/blockwise_gate.py`) for `blk200n1` within `5%` of
  the same quantity on the committed pool's rank-0 draws (de-echoed). **Power:** `blk200n1_shipped` must
  differ from the pool's by at least `10%`; if it does not, G1 is UNINFORMATIVE rather than PASS.
- **G2, the judge path.** In every pass, `u_metered_k10` and `u_anchor_k0` agree prompt by prompt with
  feat-198's pass on the same host (`results/order_averaged_h2h_per_prompt_scorer_gemma27b.csv`), which
  judged the same texts with the same judge.
- **G3.** Every arm's log prints `temperature=1.0 top_k=0 top_p=1.0` as the resolved configuration, and
  the shipped probe's prints `0.6` and `0.9`.

## Readings and predictions

Contrasts are per-prompt paired differences of order-averaged levels, `10,000` bootstrap resamples,
`95%` intervals; the verdict is the interval's side of zero.

- **B1.** Every value-scored arm gains over the anchor control (the pass's `D4` SURVIVES). *Predicted:
  all seven.*
- **B2, installments at fixed draws.** `u(blkLn64) - u(blk200n64)` for `L in {10, 25, 50}`: INSTALLMENTS WIN
  (interval above zero), ONCE WINS (below), or TIE. *Predicted: TIE at every `L`*, because the judged
  gain of whole-output selection is saturated by `n=64` (feat-129).
- **B3, installments at a fixed certificate.** `u(blk100n8)`, `u(blk67n4)`, `u(blk34n2)` against
  `u(blk200n64)`, same labels. *Predicted: ONCE WINS against all three*, since they draw `8`, `16` and `32`
  times less.
- **B4, the planner.** `blk10n64_planner` gains over the anchor (SURVIVES) and ties `blk10n64` (TIE).
  *Predicted: SURVIVES, TIE.*
- **B5, the ablation.** `blk50n64_reward` serves shorter texts than `blk50n64` (median words) and loses to
  it (`u(blk50n64) - u(blk50n64_reward)` above zero). *Predicted: shorter, and it loses.*
- **B6, leakage.** `blk10n64_memoriser`: mean and maximum near-verbatim recall `0.0000` over the `100`
  passages. *Predicted: both zero.* ROUGE-L is reported against the whole-output adversarial arm's at
  `n=64` on the same passages (`results/selection_extraction_paraphrase_per_passage.csv`, `rouge_n64`,
  mean `0.1073`) without a band.
- **Descriptive, no band.** Served length and empty count per arm; the null arm `blk200n1` against the
  anchor control; each arm against the committed `sel_n64` (the pass's `D5`); reward calls and draw
  tokens per arm.

## What the manuscript does with each outcome, fixed now

Section 3's paragraph on what the dichotomy does not cover reports the measurement beside the
certificate it states, with a table in the appendix, and Limitations drops "we did not measure
selection in installments" whatever the arms read.
- If any `L` in B2 reads INSTALLMENTS WIN: the paragraph says installments buy more than one choice
  at an informative window certificate, and the conclusion stops describing the budget as spent once.
- If B2 reads TIE or ONCE WINS throughout: the paragraph says that at this scorer and anchor,
  installments buy nothing over one choice, so the growth of their certificate with the number of
  blocks is unpaid for.
- If any arm in B3 ties or beats once: the compute paragraph says a certificate of `log 64` is reached
  at a fraction of the draws. If all lose, the appendix says so.
- B4 is reported beside the whole-output likelihood ranking (`-0.006`), B6 beside the whole-output
  adversarial arm.
- If G1 fails, no judged number from this arm is quoted.

## Excluded in advance

- Any other `L`, `n`, scorer, rollout count or judge chosen after a draw is read; judging on another
  host; pooling arms; re-running an arm with a new seed because of its result.

## Scoring log
