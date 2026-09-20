# Pre-registration: is the metered decoder's TriviaQA certificate vacuous, and at which budget?

Written **before** `analysis/tqa_vacuity.py` exists and before any number is computed.
Nothing above the `## Scoring log` heading is edited afterwards.

## Why this arm exists

`sections/appendix_selection.tex` contradicts itself inside one paragraph. It says

> At the budgets whose certificate is not vacuous it gains nothing: $0.112$ at $k=0$ and
> $k=0.5$ ($12$ nats), $0.114$ at $k=1$ ($24$ nats) and $0.166$ at $k=3$ ($72$ nats, already
> vacuous)

and then closes with

> No certificate, leakage, vacuity or $s(x)$ number comes from this arm.

Both cannot stand. The second sentence is the true one today: **no `S(x)` for a TriviaQA answer is
measured anywhere in `results/`.** `regimes_*.csv` measures per-character surprisal on protected
prose, not on short factual answers, so `already vacuous` and `the budgets whose certificate is not
vacuous` are unsourced adjectives about a set of numbers --- caution (ai), and the contradiction
inside one file is caution (ao). This arm measures the missing quantity so the sentence can be a
measurement or be withdrawn.

## The quantity

Proposition~2's threshold is order-invariant vacuity at `K >= S(x)`, where `S(x)` is the **anchor's**
surprisal of the target event. Here the event is *the served model emits an acceptable answer*, so

    S(x) = -log P_anchor(any accepted alias | served prompt)

and the served prompt is `prompt_text` from `data/bench/triviaqa_factual.jsonl` verbatim --- the same
string `h1.py` served the metered arm, read from the same file, never reconstructed (caution (aa)).
Accepted aliases are the ` ||| `-separated `reference` field.

`P(event) >= max_alias P(alias)`, so this arm scores `S(x) = -log max_alias P(alias)`. That
**understates** `P(event)` and therefore **overstates** `S(x)`, which is conservative **against**
the claim under test: a larger `S(x)` makes `72` nats harder to call vacuous. Stated here so the
direction cannot be chosen later.

Budgets on this arm are `K = k T_max` at `T_max = 24`: `k=0.5 -> 12`, `k=1 -> 24`, `k=3 -> 72`,
`k=20 -> 480` nats.

## Instrument gates, read FIRST and before any H1--H3 number is computed

Caution (au), written this session after a third arm died of a parser never tested against real
output: **the instrument is checked before the result, and the check is run on observed output.**

- **G1 (format).** Before the full run, five items are scored with the continuation's tokenization
  and per-token surprisal PRINTED and read --- specifically that the leading space after
  `Answer:` is part of the first continuation token and is not double-counted or dropped. This
  reveals format, not accuracy, so it is not peeking.
- **G2 (the better model must be less surprised).** The risky model `Llama-3.1-8B-Instruct` scores
  `0.618` on this task against the anchor's `0.112`. Its median `S(x)` on the same gold answers
  under the same prompts **must be strictly below** the anchor's. If it is not, the scorer is
  wrong --- wrong prompt, wrong alias handling, wrong tokenization --- and the arm is **INVALID**,
  not a result.
- **G3 (the prompt must matter).** Permutation control: re-score each item's anchor `S(x)` against
  a *different* item's gold aliases. Median permuted `S(x)` **must exceed** median true `S(x)` by
  at least `5` nats. If it does not, the model is not conditioning on the question and the arm is
  **INVALID**.

## Predictions, with the refuting observation for each

- **H1 (what licenses `already vacuous` at `k=3`).** The fraction of the 500 questions with
  `S(x) <= 72` is **at least `0.50`**. REFUTED if below `0.50`, in which case `already vacuous`
  is withdrawn from the manuscript.
- **H2 (what licenses calling `k=1` a non-vacuous budget).** The fraction with `S(x) <= 24` is
  **below `0.50`**. REFUTED if at or above `0.50`, in which case the phrase `the budgets whose
  certificate is not vacuous` is false as written and the sentence is rewritten as a per-budget
  vacuous fraction.
- **H3 (`k=0.5`).** The fraction with `S(x) <= 12` is **below `0.25`**. REFUTED if at or above.
- **H4 (`k=20`, the meter's winning arm).** The fraction with `S(x) <= 480` is **at least `0.95`**.
  REFUTED if below. This is the arm the paper already calls a non-decoder; if its certificate is
  not vacuous on nearly every question, that characterisation needs revisiting.

## What this arm may and may not change

It may source, rewrite or withdraw the vacuity adjectives in that one paragraph, and it may add a
per-budget vacuous fraction. It may **not** touch any accuracy, gain, cost or leakage number: no
generation is produced and nothing is re-judged. If G2 or G3 fails the arm is INVALID and the
adjectives are **withdrawn** rather than kept unsourced --- a defect in our instrument does not
license leaving the contradiction standing.

## Scoring log

(to be filled after the run)
