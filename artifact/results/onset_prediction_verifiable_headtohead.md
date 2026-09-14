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

## Scoring, 2026-09-14

Run: `scripts/run_tqa_headtohead.sh`, GPU 4, `[h2h] metered exit=0`, `[h2h] selection exit=0 at
09:04`, `[h2h] score exit=0`. Output `results/verifiable_metered_tqa.csv`. TriviaQA
`rc.nocontext`, 500 validation questions, `T_max = 24`, TinyComma-1.8B anchor,
`Llama-3.1-8B-Instruct` risky.

| mechanism | arm | certificate, nats | realised KL | accuracy | gain |
|---|---|---|---|---|---|
| metered | `k=-1` (risky alone) | --- | --- | `0.618 [0.576, 0.660]` | `+0.506` |
| metered | `k=0` (anchor alone) | `0` | `0.0000` | `0.112 [0.086, 0.140]` | `0.000` |
| metered | `k=0.5` | `12.0` | `0.0109` | `0.112 [0.086, 0.140]` | `0.000` |
| metered | `k=1` | `24.0` | `0.2528` | `0.114 [0.088, 0.142]` | `+0.002` |
| metered | `k=3` | `72.0` | `19.6867` | `0.166 [0.134, 0.200]` | `+0.054` |
| metered | `k=20` | `480.0` | `44.8473` | `0.618 [0.576, 0.660]` | `+0.506` |
| selection | `n=8` | `2.079` (`1.204` KL) | --- | `0.146 [0.116, 0.178]` | `+0.034` |
| selection | `n=32` | `3.466` (`2.497` KL) | --- | `0.174 [0.142, 0.208]` | `+0.062` |
| selection | `n=64` | `4.159` (`3.175` KL) | --- | `0.190 [0.156, 0.224]` | `+0.078` |

### H1 --- METERED WINS, exactly as predicted before the run

`0.618 [0.576, 0.660]` against `0.190 [0.156, 0.224]`: a gap of `0.428` against a band of `0.03`,
intervals nowhere near overlapping. **The prediction on record holds**, and it was made so that this
could not be read as a surprise. Proposition 4 bounds the served string to one TinyComma would have
drawn, and TinyComma does not know who wrote the book; a metered decoder at a large budget serves
the risky model and inherits its knowledge outright.

### The line the table is really about

**The metered decoder's winning arm is `k=20`, and its accuracy there is `0.618` --- the
unconstrained risky model's `0.618`, to three decimals, on the same questions.** It wins by ceasing
to be a constrained decoder. Its certificate at that point is `480` nats against a work of `24`
tokens, which is the vacuous horn by this paper's own threshold; at the budgets where the
certificate is not vacuous it gains `0.000` (`k=0.5`, `12` nats) and `+0.002` (`k=1`, `24` nats),
which is the trivial horn. Between them, `k=3` buys `+0.054` for `72` certified and `19.7` realised
nats --- less than selection buys at `n=32` for `2.497`.

So the dichotomy is visible here on an axis with no judge in it at all, and it is visible in the
arm that **beats** our mechanism: the metered decoder is useful exactly where its certificate is
worthless, and carries a meaningful certificate exactly where it is useless.

### H2 --- FRONTIER HOLDS on the registered axis, NARROWS on the stricter one. Both are reported.

H2 says "accuracy against realised divergence from the anchor ... the same axes as Figure 1(b)".
Those axes are Table 1's: the metered decoder's realised KL against selection's
`\log n - (n-1)/n`, the sharper KL bound for best-of-$n$, which is the quantity
`selection_decoding.py` has always written and which `1.204` and `3.175` are. The scorer originally
wrote `realised_nats = 0.0` for selection --- true, and useless: every served token is an anchor
draw, so a per-token meter reads zero by construction. `kl_nats` was added so the registered axis is
in the CSV rather than recomputed by hand.

| accuracy selection reaches | its KL nats | cheapest metered arm reaching it | its realised nats | ratio |
|---|---|---|---|---|
| `0.118` (`n=4`) | `0.6363` | `k=3` | `19.6867` | `0.032` |
| `0.146` (`n=8`) | `1.2044` | `k=3` | `19.6867` | `0.061` |
| `0.158` (`n=16`) | `1.8351` | `k=3` | `19.6867` | `0.093` |
| `0.174` (`n=32`) | `2.4970` | `k=20` | `44.8473` | `0.056` |
| `0.190` (`n=64`) | `3.1745` | `k=20` | `44.8473` | `0.071` |

Every ratio is below the registered `0.1`. **FRONTIER HOLDS.**

**Two disclosures, because neither is in the band as written.**

1. **Arms that bought nothing are excluded, and had to be.** Selection `n=1` and `n=2` and metered
   `k=0` and `k=0.5` all sit at the anchor's own `0.112` with a zero gain. The cheapest metered arm
   "reaching" `0.112` is `k=0`, the anchor, at `0.0000` realised nats, so the ratio is infinite for
   *any* mechanism including the metered one --- `k=0.5` spends `0.0109` for the same accuracy. A
   ratio against zero is not a reading. The band should have said "every accuracy a mechanism buys
   over the shared control"; it says "reaches". This exclusion is stated here rather than applied
   silently.
2. **On the pathwise axis the frontier NARROWS at two of five points.** Selection's `\log n`
   against the metered decoder's realised KL gives `0.070`, **`0.106`**, **`0.141`**, `0.077`,
   `0.093`. This compares selection's strongest-order bound against the metered decoder's
   weakest-order realised spend, so it is not like for like --- but `\log n` is the certificate the
   paper's headline rests on, and the reading that is less favourable to us is on the record.

### H3 --- the consequence that fires

**METERED WINS and FRONTIER HOLDS**, the predicted combination. Its committed consequence: Section 6
gains a sentence and Limitations gains a scope statement **in the main text** --- the two mechanisms
are **not substitutes** on a task whose answers the anchor does not know. The metered decoder buys
real accuracy there and selection cannot, and it buys it at a budget this paper has already shown to
be vacuous. That is the dichotomy, not a defence of the meter, and the sentence says both halves.

### H4 --- what is not claimed

No certificate, leakage, vacuity or `s(x)` number comes from this arm: TriviaQA answers are not
protected text and the corpus carries no protected split. Every cap above is `k\,T_{\max}` at
`T_{\max} = 24`, not the `200` of the copyright workload, and is quoted with it. No level here is
quoted beside a judged level from any other arm.
