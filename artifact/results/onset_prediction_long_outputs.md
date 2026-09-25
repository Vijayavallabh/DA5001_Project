# Pre-registration: selection, installments and the meters at 1,000-token outputs (feat-213)

**feat-213.** Committed **2026-09-25**, after a smoke of every command below on one prompt per class
(`output/feat213_smoke`, host B GPUs 0-2) and before any token of the arms below. Nothing above
`## Scoring log` is edited after the first token.

## Why

Review 3 (Q12): "For outputs of 1,000 tokens or more, does whole-output selection still buy utility, or does
it require installments whose certificate grows linearly in T? If the latter, how does it compare with the
meter at matched nats?" Every judged arm on record stops at `T_max = 200`.

## What runs (host B, GPUs 0-3; `scripts/run_feat213.sh`)

The headline's `500` prompts, TinyComma-1.8B with Llama-3.1-8B-Instruct, temperature `1.0`, no penalty, no
chat template, **`T_max = 1000`**.

- **Meters and opponent** (`h1.py --trajectories-per-prompt 1 --seeds 61 62 63 --cap-neutral 200
  --cap-factual 150 --cap-creative 150 --cap-val 0 --cap-test 0 --cap-attack-train 0 --batch-size 48
  --trust-remote-code --max-new-tokens 1000`):
  - `kl`: `--k-values -1 0 0.020794 0.1 0.5 10`. This is He et al.'s KL meter with its prefix debt. `k=-1` is the
    opponent, `k=0` the anchor, and `0.020794` gives `K = 20.79` nats, the installments' whole-output
    certificate (a weaker order). `0.1` is the authors' smallest `k` (`K = 100`).
  - `pw`: `--constraint pathwise --no-prefix-debt --k-values 0.0041589 0.020794`, the per-token pathwise meter
    at whole-output `log 64 = 4.16` and `20.79` nats.
  - `front`: `--constraint pathwise --no-prefix-debt --initial-bank 4.158883 --k-values 1e-9`, which spends
    `log 64` up front.
  - `win`: `--constraint pathwise --window 50 --no-prefix-debt --k-values 4.1589`, the sliding-window meter at
    `log 64` per `50`-token window.
- **Selection** (`analysis/blockwise_selection.py`, the anchor alone, pure sampling):
  - `sel`: `--block-len 1000 --t-max 1000 --n 64 --scorer reward --reward-max-chars 0 --pool-arms 1 4 16 64`.
    This is whole-output best-of-`n` over one pool of `64` draws, at `n = 1, 4, 16, 64`. `n=1` is the
    pool's rank-0 draw and serves as the paired control. Certificate: `log n`, whole output and every window.
  - `inst`: `--block-len 100 --t-max 1000 --n 8 --scorer value --reward-max-chars 0`. These are installments,
    `8` draws per `100`-token block scored by the committed value rule. Certificate: `10 log 8 = 20.79`
    nats for the whole output and `2 log 8 = 4.16` for any `50`-token window.
  - `--reward-max-chars 0` is the committed reward (same model, same template) with its `1,200`-character cut
    removed. At `T_max = 200` the cut seldom binds. At `1000` it would hide everything after about the
    first `270` tokens, and in installments every block after the third would be scored identically.
- **Judging.** `analysis/matched_h2h.py`, both presentation orders, de-echoed text, against the `T_max = 1000`
  opponent (`--baseline-dir output/feat213/kl`, lowest seed), **uncut** (`--judge-max-chars 0`). A pair
  whose judge prompt exceeds `8,180` tokens has both texts cut to the longest `500`-character multiple that
  fits, in both orders (`--fit-window 8180`), and the count is reported. There are two passes, both on host
  B and each with a fresh verdict cache: judge~B (`Phi-3.5-mini-instruct`, tag `f213_B`) and judge~G
  (`gemma-2-27b-it`, tag `f213_G`).
  - Arms: `sel_n64`, `sel_n16`, `sel_n4`, `sel_n1`, `inst`, `anchor_k0`, `kl_20.8`, `kl_0.1`, `kl_0.5`,
    `kl_10`, `pw_4.16`, `pw_20.8`, `front_4.16`, `win_4.16`.
  - Controls: every `sel_*` and `inst` against `sel_n1`, and every meter against `anchor_k0`.

## Predictions (judge~B unless named)

- **L1** Whole-output selection still buys utility at `1,000` tokens: gain of `sel_n64` **CONFIRMED**.
- **L2** Installments minus the pathwise meter at the same whole-output `20.79` nats (`inst - pw_20.8`):
  **CONFIRMED**.
- **L3** Installments minus He et al.'s KL meter at the same `20.79` nats, a weaker order (`inst - kl_20.8`):
  **CONFIRMED**.
- **L4** Whole-output selection minus each meter at its own certificate `log 64` (`sel_n64 - pw_4.16`,
  `sel_n64 - front_4.16`, `sel_n64 - win_4.16`): each **CONFIRMED**.
- **L5** Whole-output selection minus the KL meter at `k = 0.1` (`K = 100`) and at `k = 0.5` (`K = 500`): each
  **CONFIRMED**.
- **L6** Judge~G gives L1-L5 the **same sign**.

**Descriptive, no band.**
- the gains at `n = 4, 16` (the `T_max = 1000` scaling curve);
- the installments' gain, and `inst - sel_n64` (whether installments are needed);
- `sel_n64 - kl_10`;
- each arm's mean length in tokens and words, and the share of its outputs longer than `200` tokens;
- the L1 gain restricted to prompts whose served `n=64` output is longer than `200` tokens;
- the pairs cut to fit, empties served, and order consistency per arm.

## What the manuscript does with each outcome, fixed now

An appendix paragraph and table carry every row, and Section 3's installments concession or the
Limitations sentence answers Q12 with L1's reading.
- If L1 holds, the paper says whole-output selection still gains at `1,000` tokens at the same `log 64`.
- If L1 fails, the paper says long outputs need installments, whose certificate grows as `(T/L) log n`.
- L2-L5 are reported as the matched comparison at `1,000` tokens, wherever each lands.

## Excluded in advance

- Any other `T_max`, `n`, `L`, `k`, seed, temperature, template, judge, reward cut or fit rule chosen after a
  token is decoded or a verdict is read.
- Pooling with any `T_max = 200` pass, or setting a level from this pass beside one from another.

## Scoring log

### Execution notes, 2026-09-25, written before any verdict of this pass was read

- **Two re-runs from scratch, at a smaller draw batch.** Installments (`q1`) ran out of GPU memory at block 5
  of 10, before writing any output. At the default `--gen-batch 256` the anchor's KV cache for 256 rollouts at
  ~1,100-token contexts sits beside the resident 7B scorer. The whole-output pool (`q0`) was stopped before
  writing any output, because it runs its batches in ascending prompt length and its last batches would not
  have fit either. Both were re-run from scratch at `--gen-batch 128` (`q1b`, `q0b`). Batch size moves the
  sampled draws (caution (u)), so these are the arms' only draws; nothing from the stopped runs was kept or read.
- **The judge-G pass moved cards.** While the pool ran, the user's own vLLM server started on GPUs 1 and 3,
  and `q0b` would have put judge~G on GPUs 0 and 1. Its queue shell was stopped with the pool's python left
  running. `~/v/f213_g2.sh` then waits for that python by PID, checks the pool's outputs, writes the sentinel
  the judge-B pass waits on, and runs the registered judge-G command on GPUs 6 and 7. To save time it first
  judges the ten arms that already exist under the same tag, then runs the full registered pass, which reuses
  those cached verdicts. `matched_h2h.py` judges each arm alone against the opponent, in batches of 200 in
  prompt order, so the split does not change a verdict.

### Scored 2026-09-25 (host B, judge~B and judge~G) --- four of six registered readings right

Command, on host B where `output/feat213` lives: `.venv/bin/python analysis/score_feat213.py` ->
`results/long_outputs_scoring.csv` and `results/long_outputs.csv` (lengths). No pair needed the fit rule:
`0` of `500` were cut in every arm under B. Verdicts: `results/matched_h2h_verdicts_f213_{B,G}.csv`.

| | judge~B | registered | verdict |
|---|---|---|---|
| L1 best-of-`64` over its own draw at `T_max = 1000` | `+0.063` `[+0.043, +0.083]` CONFIRMED | CONFIRMED | RIGHT |
| L2 installments minus the pathwise meter at `20.79` | `+0.0165` `[-0.0045, +0.038]` unresolved | CONFIRMED | **WRONG** |
| L3 installments minus the KL meter at `20.79` | `+0.019` `[-0.001, +0.0395]` unresolved | CONFIRMED | **WRONG** |
| L4 best-of-`64` minus the pathwise, up-front and windowed meters at `log 64` | `+0.062`, `+0.0305`, `+0.0445`, all CONFIRMED | CONFIRMED | RIGHT |
| L5 best-of-`64` minus the KL meter at `k = 0.1` and `0.5` | `+0.0775`, `+0.104`, both CONFIRMED | CONFIRMED | RIGHT |
| L6 judge~G, same sign on L1-L5 | all eight positive and CONFIRMED (`+0.1605` and `+0.152` for L2, L3) | same sign | RIGHT |

**Descriptive.**
- **Whole-output selection does not need installments at `1,000` tokens.** Installments of `8` draws per
  `100`-token block gain `+0.0185` `[-0.0005, +0.037]` under B (`+0.1525` under G) and lose to one choice
  among `64` by `0.0445` `[0.026, 0.0635]` (G: `0.059` `[0.0295, 0.0895]`). At `T_max = 200`, with `64`
  draws per block, installments won instead.
- **Lengths.** The opponent, the risky model continuing text, always runs to `1,000` tokens. Every anchor-based
  arm stops far sooner. The anchor averages `179` tokens, best-of-`64` `203` (`32.6%` of outputs longer than
  `200` tokens, `2.2%` at the cap) and installments `159`. The meters lengthen with their budget: `215` at
  `k = 0.0208`, `461` at `0.1`, `831` at `0.5`, and the cap at `10`.
- On the `163` prompts whose served best-of-`64` output is longer than `200` tokens, its gain is `+0.092`
  `[+0.057, +0.127]` (G: `+0.1825`).
- **At this length the meters buy nothing over the anchor under B.** The KL meter reads `-0.0145` at
  `k = 0.1`, `-0.041` at `0.5` and `-0.0175` at `10`, where it is the risky model. The pathwise meters read
  `+0.001` and `+0.002`. The windowed meter reads `+0.0185` and the up-front one `+0.0325`. The judges part on
  the risky model's own `1,000` tokens: G reads `k = 10` at `+0.1095` over the anchor, B at `-0.0175`.
- Selection's scaling at `1,000` tokens (B): `+0.024` at `n = 4`, `+0.0305` at `16`, `+0.063` at `64`. The
  uncut reward serves many empty answers, `97` of `500` at `n = 64` and `167` at `n = 16`, against `65` for
  one draw (`results/matched_h2h_f213_B.csv`, `n_empty`).

**What the manuscript does (as fixed above).** L1 held, so the paper says whole-output selection still gains
at `1,000` tokens at the same `log 64`. L2 and L3 are reported as the matched comparison at `1,000` tokens:
installments tie the meters at their whole-output `20.79` nats under B, beat them under G, and lose to
whole-output selection under both.
