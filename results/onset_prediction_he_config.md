# Pre-registration: the head-to-head at the authors' own pair, TinyComma + Llama-3.1-70B base (feat-185)

**feat-185.** Committed **2026-09-24**, before any judge call below. Nothing above `## Scoring log`
is edited after the first judge call.

## Why

A referee report asks for the metered decoder "as published": He et al. pair the TinyComma anchor
with **base** risky models (the 70B, Qwen 2.5 72B, Llama 4 Scout), and TinyComma + Llama-3.1-70B is
the one pair they run at the token level. Every judged head-to-head in this paper uses the released
logs' `Llama-3.1-8B-Instruct`, which feat-184 (`results/onset_prediction_served_opponent.md`) shows was
served without its chat template. A base risky model needs no template, so the authors' own pair
removes that confound entirely.

The generations already exist: `output/phase5/imit_llama70b`, TinyComma with
`unsloth/Meta-Llama-3.1-70B`, the headline's `500` prompts, one trajectory each, `k in {-1, 0, 0.5,
1, 3, 20}`, `T_max = 200`, temperature `1.0` (the harness default, not the authors' `0.7`). Their
binding rates over the three ordinary classes, measured before this registration: `k=0.5` `26.5%`,
`k=1` `6.29%`, `k=3` `0.45%`, `k=20` `0.00%`. Rate-matching to the committed target of `8.008%` over
that grid picks `k=1` (`|0.0629 - 0.0801| = 0.017`, the argmin; `k=0.5` and `k=3` bracket it). And
`k=0.5` is the only budget on the grid whose certificate says anything about a `50`-token window:
`K = 100` nats against that window's `159.8`, where every `k >= 0.8` is vacuous.

## What runs

A judge pass, no generation. `analysis/order_averaged_h2h.py --deecho`, judge B, seed `7717`, both
presentation orders, the **committed opponent** (`output/sweep_plain` `k=-1`, lowest seed) so that
the construction is the committed headline's:

- **Run C-a**: selection `n=64` and `n=1` (committed), the 70B meter at `k=20` as the metered arm, its
  control the anchor alone from the same run (`imit_llama70b` `k=0`), and the 70B meter at `k=1` as
  the extra arm. `--tag he70b_k20`.
- **Run C-b**: the same selection arms, the 70B meter at `k=0.5` as the metered arm, the same control,
  and the unconstrained 70B base (`imit_llama70b` `k=-1`) as the extra arm. `--tag he70b_k05`.

## Gates

- **G0.** All arms cover the headline's `500` prompts and the de-echo recovers at least `99%` of
  records in each.
- **G2.** Selection's per-prompt levels are identical in C-a and C-b and equal to feat-184 Part A's
  (same text, same opponent, greedy judge).

## Bands, on the committed construction (`D3` = selection's gain minus the meter's, paired)

**C1 --- vacuous budget, `k=20`, where the meter is the 70B base.** The script's own reading of `D3`:
**CONFIRMED** (interval above zero), **REFUTED** (below), **UNRESOLVED**. We predict **REFUTED**: a
70B base model continuing the prompt is a far stronger text model than the anchor's best-of-64.

**C2 --- rate-matched binding budget, `k=1`, still vacuous for a `50`-token window.** The extra arm's
gain minus selection's (`D5`): the script reads **INCUMBENT WINS** (interval above zero) /
**INCUMBENT LOSES** (below) / **TIE**. We predict **INCUMBENT WINS**: `94%` of its steps are the 70B.

**C3 --- the one informative budget, `k=0.5`.** `D3` at `k=0.5`, same readings as C1. We predict
**CONFIRMED**: at `k=0.5` the 8B pair's decoder is inseparable from its own anchor under both judges,
and the 70B pair binds harder there.

**C4 --- descriptive.** The unconstrained 70B base's gain over the anchor (Run C-b's extra arm) and
its difference from selection.

## What the manuscript does with each outcome, fixed now

The four readings go into Section 4 and Appendix H as the head-to-head at the authors' own pair, with
temperature `1.0` stated. If C1 or C2 reads against selection and C3 for it, the paper says so in one
sentence: the meter wins where its certificate is vacuous and loses where it says something. Any
other pattern is reported as measured and the paper's judged claim is scoped to it.

## Excluded in advance

- Another opponent, judge, seed or budget chosen after a judge call.
- Pooling with the 8B pair's passes, or setting these levels against another pass's.
- Reading C3 as a statement about budgets below `0.5` or about another window length.

## Compute

Two judge passes, `~5,000` Phi-3.5-mini calls each, about an hour on one A100 (local GPU 4, after
feat-184's run B3).

## Scoring log
