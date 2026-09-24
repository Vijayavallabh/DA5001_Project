# Pre-registration: the head-to-head against a properly served risky model (feat-184)

**feat-184.** Committed **2026-09-24**, before the new generation arm starts and before any judge
call below. Nothing above `## Scoring log` is edited after the first judge call.

## Why

A referee report asks whether the risky model is served at full strength, and it is not. Two facts
found while answering it, both read off the committed trajectories rather than inferred:

1. **No chat template.** `dap/e1.py` defaults to `use_chat_template = False`, and every judged arm
   behind the headline --- the fixed opponent (`output/sweep_plain` `k=-1`), the metered decoder at
   `k=10` (`output/phase2/conc_all`), its anchor control (`output/sweep_plain` `k=0`) --- records
   `chat_template = False`. `Llama-3.1-8B-Instruct` is therefore served as a raw text continuer of
   `Complete the prefix:\n<prompt>` at temperature `1.0` with no repetition penalty. Served with its
   template (`output/sweep_chat`, same prompts, same seeds, same batch size) it answers as an
   assistant. The pipeline gap the appendix reports --- the same opponent at `0.6792` through our sweep
   and `0.8519` through `analysis/blocklist_decode.py --chat` on AlpacaEval --- is this.
2. **The judged text carries the prompt's tail (caution (bc)).** `_records_from_batch` sliced each
   left-padded row at its own token count, so a row with `p` pad tokens has its last `p` prompt tokens
   at the start of `generation`. Measured on the headline's 500 prompts (lowest seed): the opponent and
   the anchor control carry an echo on `73.6%` of prompts (mean `10.7` characters, max `73`), the
   `k=10` meter on `32.6%` (mean `1.1`), and the selection pool on every creative prompt (mean `32.1`
   characters). All `64` candidates of a prompt carry the **same** echo (checked on the `150` creative
   prompts: `0` prompts with two echo lengths), so selection's picks are unaffected; the judge, which
   sees the text, is not. Fixed in `dap/e1.py` and `dap/e2/evaluator.py` for every future run
   (`tests/test_left_pad_slicing.py`); existing text is recovered exactly as `full_text` minus the
   prompt (`dap.shared.served_generation`, `tests/test_served_generation.py`).

## What runs

**Part A --- the committed pass on de-echoed text.** `analysis/order_averaged_h2h.py --deecho --tag
deecho`, every other flag the committed pass's: judge B (`microsoft/Phi-3.5-mini-instruct`), the same
four arms, the same opponent, seed `7717`, both presentation orders. No generation.

**Part B --- against the risky model served with its chat template.** One new generation arm, the
metered decoder at `k=10` with the template, on the headline's `500` prompts:

```
h1.py --use-chat-template --k-values 10 --trajectories-per-prompt 1 --seeds 52 53 54
      --cap-neutral 200 --cap-factual 150 --cap-creative 150 --cap-val 0 --cap-test 0
      --cap-attack-train 0 --batch-size 48 --trust-remote-code --output-dir output/feat184/chat_k10
```

The seed tuple is disjoint from the opponent's (`42 43 44`), so the meter's draws are independent of
the opponent's rather than byte-identical to them (a `k=10` meter sharing the opponent's seed stream
reproduces it on most prompts, which is the degenerate comparison the appendix already names). Every
other setting is the committed sweep's: temperature `1.0`, no penalty, `T_max = 200`, prefix debt on.

Judged with judge B against the **opponent served with its template** --- `output/sweep_chat` `k=-1`,
lowest seed --- every item in both orders, generations de-echoed:

- **Run B1**: selection `n=64` and `n=1` (the committed arm, unchanged), the new chat meter at `k=10`,
  its anchor control (`output/sweep_chat` `k=0`), and a **second, independent draw of the opponent's
  own configuration** (`output/sweep_chat` `k=-1`, second-lowest seed).
- **Run B2**: the same selection arms, and the committed plain-protocol meter at `k=10` and its
  control, with the committed opponent's own text (`output/sweep_plain` `k=-1`) as the extra arm.
- **Run B3, descriptive, no band**: the chat meter at `k=1` (`output/sweep_chat`; it binds on `15.7%`
  of steps under the template against `6.0%` without it, because the anchor now reads chat tokens it
  never saw in training).

The statistic for Part B is the **order-averaged level** of each arm against the one opponent, and
the paired difference of two arms' levels over the `500` prompts. A gain over each arm's own control,
the committed construction, is also printed but is not read: under the template the meter's control
is the anchor reading chat tokens, a different control from selection's, so the two gains no longer
share a baseline.

## Gates, read before any band

- **G0, the corpus.** All arms and the opponent cover the same `500` prompt ids, and the de-echo
  recovers the generation on at least `99%` of records in every arm (`full_text` begins with the
  prompt, and `generation` ends with the remainder).
- **G1, the instrument (Part B).** The opponent's second independent draw reads an order-averaged
  level within `[0.40, 0.60]` against the first. Outside it, the judge cannot tell a model from itself
  and no Part B band is read.
- **G2, determinism.** Selection's `n=64` and `n=1` levels are identical, to the prompt, in B1 and B2
  (same text, same opponent, greedy judge). If not, the runs are not comparable and no cross-run
  difference is read.

## Bands

**A1 --- the committed difference on de-echoed text.** `D3` (difference of gains, paired) recomputed.
**REPRODUCES** if its interval excludes zero and `|D3 - 0.0645| <= 0.02`; **SHIFTS** if it excludes
zero and moved further; **FAILS** if the interval contains zero or lies below it. We predict
**REPRODUCES**: the echo is short and, within a prompt, shared by arms from one run.

**B1 --- the reversal against a properly served risky model.** `L = level(selection n=64) -
level(chat meter k=10)`, paired, bootstrap over prompts. **HOLDS** if the interval lies above zero,
**FAILS** if it lies below, **UNRESOLVED** otherwise. We predict **FAILS**: at `k=10` the meter is the
risky model, and served with its template the risky model answers as an assistant where the anchor's
best-of-64 continues text.

**B2 --- the committed meter against a properly served opponent.** `level(selection n=64) -
level(plain meter k=10)` against the chat opponent, same three readings. We predict **UNRESOLVED**:
this is the opponent ladder's question with the committed opponent's own model at full strength, and
every rung at or above strength `0.773` read unresolved.

**B3 --- the size of the serving handicap, descriptive.** `level(second chat draw) - level(committed
plain opponent text)`, both against the chat opponent. We predict at least `+0.15`.

## What the manuscript does with each outcome, fixed now

- **Either way**, the paper states how the risky model was served --- no chat template, temperature
  `1.0`, no repetition penalty --- in Section 4.1, that He et al. serve base models at temperature `0.7`
  with penalty `1.1`, and it stops saying the meter was measured "at its own temperature". The echo
  defect and its repair are stated in the judging protocol.
- **A1**: if `|D3 - 0.0645| < 0.005` the committed figure stays the one quoted, with the de-echoed
  figure beside it once; otherwise the de-echoed figure replaces it wherever the paper quotes the
  headline difference.
- **B1 FAILS or UNRESOLVED**: the judged head-to-head is reported as a property of the risky model's
  serving configuration, not of where the budget is spent; the abstract and introduction stop
  presenting it as selection being judged better than the metered decoder without that
  configuration; the properly-served result is given beside it in the main text.
- **B1 HOLDS**: both readings are reported and the configuration is still stated.

## Excluded in advance

- Choosing the judge, the opponent's seed, the prompts or the statistic after any judge call.
- Pooling Part A with Part B, or setting a Part B level against a committed level (different
  opponents, so different passes: caution (ap)).
- Reading a binding-budget chat meter as the published mechanism: in this harness both models share
  one context, so under the template the anchor reads chat tokens; B3 is descriptive for that reason.
- Re-running with a different temperature or penalty and reporting whichever reads better.

## Compute

Part A and Part B are judge passes (`~5,000` Phi-3.5-mini calls each, under `30` minutes on one A100);
the one generation arm is `500` prompts at `200` tokens with TinyComma and the 8B, about `20` minutes.
Well under the 24-hour threshold. Local GPUs 1, 2 and 4 (idle at registration), never GPU 3.

## Scoring log

### Scored 2026-09-24 10:20 IST --- A1 REPRODUCES (and replaces +0.0645), B1 FAILS, B2 HOLDS

All runs exited `0` (`output/logs/feat184_*.done`; launcher `scripts/run_feat184.sh`). The chat meter
at `k=10` generated in `4` minutes on GPU 2 with the fixed slicing (`output/feat184/chat_k10`; its
`generation` fields carry no echo, checked on a smoke run). Scored by
`.venv/bin/python analysis/served_opponent.py --out results` -> `results/served_opponent.csv`.

**Gates.** G0 PASS: `500` prompts in every arm, and the de-echo recovers `100.0%` of records in every
directory read (`1,500/1,500` for each sweep arm, `32,000/32,000` for the selection pool). G1 PASS:
the chat-served opponent's own second draw reads `0.504` against the first, a coin flip. G2 PASS:
selection's per-prompt levels are identical in B1 and B2.

**A1 --- REPRODUCES.** On de-echoed text the committed pass reads selection `+0.1015 [+0.0765,
+0.1260]`, the meter `+0.0510 [+0.025, +0.076]`, and `D3 = +0.0505 [+0.0155, +0.0860]`: interval clear
of zero, `0.014` from `+0.0645`. That is inside the `0.02` band and outside the `0.005` rule fixed for
what the paper quotes, so **the de-echoed `+0.0505` replaces `+0.0645`** wherever the paper quotes the
headline difference. Prediction REPRODUCES, **right**.

**Part B, levels against the chat-served opponent** (order-averaged, `500` prompts):

| arm | level |
|---|---|
| selection `n=64` (committed) | `0.3630` |
| selection `n=1` | `0.2485` |
| chat meter `k=10` (new) | `0.5010` |
| chat anchor `k=0` | `0.2380` |
| chat opponent, second draw | `0.5040` |
| plain meter `k=10` (committed) | `0.3050` |
| plain anchor `k=0` (committed) | `0.2505` |
| plain opponent text (committed) | `0.3200` |

**B1 --- FAILS**: selection minus the chat meter at `k=10` is `-0.1380 [-0.1670, -0.1070]`. Served with
its template the risky model answers as an assistant, and at `k=10` the meter is that model. Prediction
FAILS, **right**.

**B2 --- HOLDS**: selection minus the committed plain meter at `k=10`, judged against the chat-served
opponent, is `+0.0580 [+0.0325, +0.0845]`. Prediction UNRESOLVED, **wrong**, in the direction of the
paper's committed claim: the ordering of the two committed arms survives a full-strength opponent.

**B3 --- the serving handicap**: the chat-served second draw minus the committed plain opponent text is
`+0.1840 [+0.1550, +0.2130]`. Prediction at least `+0.15`, **right**.

**B3k1, descriptive (the harness gives the anchor chat tokens)**: the chat meter at `k=1`, binding on
`15.7%` of steps, reads `0.3220`; selection minus it is `+0.041 [+0.012, +0.0705]` on levels, and
`D3 = +0.0305 [-0.0045, +0.0650]` on gains.

**Two further facts the de-echoed text shows, not banded.** The audited anchor's first draw is empty on
`64` of `500` prompts (`12.8%`), not the `6.8%` the echo-carrying text implied, and `13.9%` of the
`32,000` candidates are empty. And the reward, which scored echo-carrying text, serves an empty
completion at `n=64` on `41` prompts although every one of them had a non-empty candidate: the echo
defect handicapped selection's reward, so every selection gain on record is measured with that
handicap. No prompt has all `64` candidates empty.

**Manuscript, as registered.** The paper states the serving configuration in Section 4.1 and the echo
defect in the judging protocol; the headline difference becomes `+0.0505`; B1's failure is reported in
the main text beside the committed comparison, and the abstract and introduction stop presenting the
judged comparison without its serving configuration.
