# Pre-registration: the two mechanisms compared without a judge

Committed **before either arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists, and why it did not before

Section 6 compares the two mechanisms by **judged preference**, at three pairs. The judge is
position-dominated, which is why the paper reports gains over each arm's own control and measures a
cross-pass floor of about `0.04`. Those repairs make the comparison fair; they do not make it
objective.

A judge-free comparison needs an anchor that both **shares the risky model's tokenizer** --- the
factory fuses two distributions over one vocabulary, and TinyComma is the only openly licensed safe
model that does --- and **can do the task**. `results/onset_prediction_verifiable.md` had to record
that TinyComma scores `0.04` on GSM8K and therefore that *"there will be no judge-free head-to-head
against the metered decoder"*. On TriviaQA it scores `0.07`, above the same gate. The obstacle was
the task, not the design, and this arm removes it.

## What is run

Both mechanisms from `h1.py` on `data/bench/triviaqa` --- the same 500 questions, the same 5-shot
prompt, the same `24`-token cap, the same seeds --- so neither gets a code path the other does not:

```
metered   --k-values -1 0 0.5 1 3 20  --trajectories-per-prompt 1
selection --k-values 0                --trajectories-per-prompt 64
```

Anchor `jacquelinehe/tinycomma-1.8b-llama3-tokenizer`, risky model
`meta-llama/Llama-3.1-8B-Instruct`. `k = -1` (risky alone) and `k = 0` (anchor alone) are the
mandatory baselines and `k = 0` is also selection's `n = 1`. Selection's rule is **majority vote**,
which returns one of the `n` draws and therefore carries the same `\log n` pathwise certificate.

Scoring is alias containment on the answer line, identical for every arm, as fixed in
`results/onset_prediction_verifiable_triviaqa.md`.

## Bands, committed before the run

**H1 -- absolute accuracy. Which mechanism serves better answers?** The metered decoder's best arm
against selection's best.

| reading | band |
|---|---|
| METERED WINS | the metered decoder's best accuracy exceeds selection's best by more than `0.03`, intervals not overlapping |
| TIE | the two are within `0.03` |
| SELECTION WINS | selection exceeds the metered decoder's best by more than `0.03`, intervals not overlapping |

**We predict METERED WINS, and predict it here so the arm cannot be read as a surprise.**
Proposition 4 bounds the served string to one the anchor would have drawn, so selection cannot
supply a fact TinyComma does not have, while a metered decoder at a large budget is serving the
risky model and inherits its knowledge outright. **If the prediction holds, the paper owes a scope
statement it does not currently make** (see H3). If SELECTION WINS or TIE, the prediction is wrong
and the constructive claim is stronger than the paper currently argues.

**H2 -- the frontier. What does each pay for what it buys?** Accuracy against realised divergence
from the anchor, in nats, for every arm of both mechanisms --- the same axes as Figure 1(b), with
exact match in place of the judge.

| reading | band |
|---|---|
| FRONTIER HOLDS | at every accuracy selection reaches, it reaches it at less than a tenth of the nats the cheapest metered arm reaching that accuracy spends |
| FRONTIER NARROWS | the ratio is between `0.1` and `1` |
| FRONTIER INVERTS | some metered arm reaches an accuracy selection reaches, at fewer nats |

H2 is the comparison the paper actually makes --- `171.3` against `3.175` is a statement about nats,
not about levels --- so H2 is primary and H1 is the scope statement beside it.

**H3 -- the manuscript consequence, fixed now for every combination.**

- **METERED WINS and FRONTIER HOLDS** (the predicted outcome). Section 6 gains a sentence and
  Limitations gains a scope statement in the main text: the two mechanisms are **not substitutes**
  on a task whose answers the anchor does not know. The metered decoder buys real accuracy there and
  selection cannot, and it buys it at a budget whose certificate this paper has already shown to be
  vacuous or trivial --- which is the dichotomy, not a defence of the meter.
- **METERED WINS and FRONTIER NARROWS or INVERTS.** The nats-for-utility claim is task-dependent and
  Section 6 must say so, naming TriviaQA and the ratio. This weakens the paper's central comparison
  and the sentence is owed regardless.
- **TIE or SELECTION WINS.** The paper may state that the reversal holds off the judge's axis as
  well, and the GSM8K arm's "not an artefact of the judge" becomes a claim about the *comparison*
  and not only about the gain.

**H4 -- what may not be claimed.** No certificate, leakage, vacuity or `s(x)` number comes from this
arm: TriviaQA answers are not protected text and the corpus carries no protected split. The metered
decoder's published cap here is `k\,T_{\max}` at `T_{\max} = 24`, not the `200` of the copyright
workload, and any cap quoted must carry its `T_{\max}`.

## Excluded alternatives

- Changing the budget grid, the shot count, the cap, the question slice or the containment rule
  after seeing a number.
- Reporting H1 without H2, or either without both mandatory baselines.
- Reading a metered win as evidence that its certificate means anything. The two are independent and
  the paper's claim is about the certificate.
- Substituting a larger anchor to make selection competitive. TinyComma is the only anchor at which
  the metered decoder runs, which is the whole reason this comparison did not exist.
- Quoting any level from this arm beside a judged level from any other.

## Scoring log
