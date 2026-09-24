# feat-159 --- Evaluation parity: give the METERED decoder the same reward model

Committed **before any generation**. Nothing above `## Scoring log` is edited afterwards.

## Why

This is the one point in the Program Chairs' report the paper answers in pieces and never by
measurement. The report says:

> The baseline metered decoder optimizes utility by fusing per-token probabilities, whereas
> selection anchoring uses a separate, explicit sequence-level reward model (or majority vote).
> This introduces a potential asymmetry in the evaluation [...] the reported utility gains for
> selection anchoring appear to stem largely from the introduction of the external reward model's
> alignment signal, an advantage the baseline metered decoder does not possess.

The paper's existing answers are real but indirect: the risky model's own log-likelihood as the
scorer **fails** (`-0.006` at `n=8`), majority vote needs no reward model at all and carries the
judge-free headline (`0.320 -> 0.546` on GSM8K), and at matched compute selection **loses**
(`-0.0395`). None of them does the obvious thing, which is to **give the meter the same signal**.

**What the arm is really about, and it is a statement about budgets rather than about scorers.**
Best-of-`n` over any decoder `q_K` with sequence budget `K` gives `q(y) <= n q_K(y)` by the same
union bound as Proposition~1, so `D_inf(q || p_s) <= K + log n`: **the budgets add**. Parity is
therefore achievable and it is not free. The question this arm answers is how much utility the
extra `K` nats buy over spending `log n` alone, on a task where the meter already wins.

## What runs

`h1.py` on `data/bench/triviaqa`, anchor `TinyComma-1.8B`, risky `Llama-3.1-8B-Instruct` --- the
one pair that shares a tokenizer and is therefore the only meterable one, the same pair as the
committed judge-free head-to-head --- with `--trajectories-per-prompt 16`, one `k` per card:

| card | `k` | certificate | what the committed arm at this `k` does | role |
|---|---|---|---|---|
| 4 | `0.5` | `12` nats | `0.112`, realised `0.0109` nats: the meter **is** the anchor | near-control: the parity arm should land near selection's own numbers |
| 5 | `1` | `24` nats | `0.114`, realised `0.2528` nats: still essentially the anchor | second near-control |
| 6 | `3` | `72` nats | `0.166`, realised `19.6867` nats | the genuinely constrained middle, and the budget the mechanism's authors publish |
| 7 | `20` | `480` nats | `0.618`, realised `44.8473` nats: the meter **is** the risky model | the strongest metered arm |

Every one of the `16` draws is then scored by **the same pointwise reward every selection arm in
this paper uses** --- `Qwen2.5-7B-Instruct`, `log p("Yes") - log p("No")` on the one fixed template,
through `score_rewards()` in `analysis/selection_verifiable.py` --- and the argmax is served.
Scored by the committed TriviaQA alias containment, `extract_tqa` + `correct_tqa`, unchanged.
**Neither mechanism gets a code path the other does not**, which is the same rule
`verifiable_metered.py` was built under.

The selection side is **re-used, not re-drawn**: `output/phase5/tqa_sel64` already holds `64` anchor
draws on these same `500` questions, and the reward-selected arm over them is computed with the
identical scorer in the identical call.

**Generations are read before this file is finalised** (caution (au)). Four metered completions at
each of `k = 0.5, 3, 20` were printed verbatim and passed through `extract_tqa`: the shape is
`': Amanda Bitton'` / `': galaxy'` at low `k` and a run-on into the next few-shot block at higher
`k`, which the extractor already cuts at the first newline. No new parser is written and none is
needed. That reveals FORMAT and not accuracy; no band was seen.

**One host detail, verified rather than assumed, because G2 would otherwise fail by accident.**
The committed arm records `target_model = meta-llama/Llama-3.1-8B-Instruct`. Host B's cache holds
only the `Meta-` prefixed duplicate, so the run is pointed at the committed id through a symlink.
That is sound only if the two are the same weights, which was checked before anything was launched:
**all four `safetensors` shards are md5-identical** (`b903f070...`, `349fbd03...`, `f3788976...`,
`df2e75d8...`). The substitution therefore changes the path a loader walks and nothing a GPU
computes. Recorded because an unverified symlink between two model ids is exactly the kind of
silent pipeline change caution (at) exists to catch, and because G2 passing must mean the pipelines
match rather than that a name was made to match.

## Gates, per `k`, read before that `k`'s band

**G1 --- instrument, on the UNCONSTRAINED risky model, with a DIRECTION** (caution (au)). The
`k=-1` arm must read **at or above `0.40`**. `results/verifiable_metered_tqa.csv` has it at `0.618`
on these exact questions, and a reading far below means the parser or the pipeline is broken --- the
second of caution (au)'s two tells, and the gate whose absence let two MMLU arms and one LAMBADA arm
reach a scoring log. The `0.40` floor is the one already fixed in
`results/onset_prediction_lambada_headtohead.md` and it is a defect-scale threshold, not a tuned one.

**G2 --- the pipelines must match, asserted and not hoped** (caution (at)). Every trajectory records
`target_model` and `anchor_model`; the new arms must carry the same pair as
`output/phase5/tqa_metered`. Read off both runs by the scorer. A mismatch is STRUCTURAL and is
reported as such, never under a cause-undetermined wording.

**G3 --- distributional, and deliberately NOT bit-identity.** The new arms draw `16` trajectories
per prompt where the committed arm drew `1`, so the RNG stream differs and rank `0` is **not** the
committed trajectory (cautions (u) and (v): batch size and draw count are part of the seed). A
bit-identity gate here would be incoherent rather than merely strict, which is the reason feat-131
deliberately wrote none. The check is therefore: **each arm's `n=1` accuracy must lie inside the
committed arm's own bootstrap interval for that `k`**, derived by the scorer from
`results/verifiable_metered_tqa.csv` --- for example `[0.134, 0.200]` at `k=3` --- rather than typed
into this file.

**Two things measured on host B BEFORE launch, both format and neither a band.** First, the
committed protocol was re-run there at `--trajectories-per-prompt 1` --- the control that changes
the host and nothing else --- and read `0.1580` against the committed `0.1660`, **inside** the
committed arm's own interval `[0.134, 0.200]`. So **G3 is satisfiable**, which is not a given: a
gate nothing can pass gates nothing (caution (as)), and this one was checked against a real run
rather than assumed. Second, a shape statistic moved that we have never baselined: `57` of `500`
host-B generations re-emit the question's tail before answering (`'?\nAnswer: ...'`) where the
committed draw does so `0` times. That is caution (au)'s echo, the shape that invalidated the
LAMBADA arm, so it was measured rather than waved past: an echo-aware parser that takes the text
after the last `Answer:` on line one recovers **exactly zero** additional correct answers on either
draw (`delta +0.0000`). The echoing items are wrong for reasons that have nothing to do with
parsing, the committed extractor leaves nothing on the table, and no parser change is made. The
shape difference is two legitimately different `bfloat16` draws on different silicon, which is
caution (as)'s regime, and it is recorded here so that a later reader does not rediscover it as a
defect.

## Bands

**B1 --- does the reward model help the meter at all?** Per `k`, paired `acc(16) - acc(1)` within
this pass, `n` on the grid `1, 2, 4, 8, 16`.

| interval | verdict |
|---|---|
| excludes zero above | **HELPS** |
| excludes zero below | **HURTS** --- the meter overoptimises against the proxy too |
| contains zero | **NO EFFECT** |

with the `2.0`-half-width marginality rule.

**B2 --- the parity verdict**, which is the reading the report asks for. The best metered arm over
(`k`, `n`) against the best selection arm over `n`, on the same `500` questions, each reported with
its budget.

| outcome | verdict | consequence, fixed now |
|---|---|---|
| the meter with the reward model beats selection with the same reward model | **PARITY MATTERS** | The objection is upheld and the paper says so plainly: a reward model helps the meter too, Section~4's comparison gave selection a signal the meter did not have, and what selection buys is the **budget** --- `log n` against `K + log n` --- and not exclusive access to a scorer. The main text carries the concession, not only the appendix. |
| it does not | **PARITY DOES NOT MATTER** | The objection is answered empirically: the extra signal does not transfer, so the gain is the placement. |
| it wins only at budgets already conceded vacuous | **PARITY AT A VACUOUS BUDGET** | Reported as is, with `results/onset_prediction_tqa_vacuity.md`'s measured median `S(x) = 5.86` nats printed beside it, since every `k` on this grid certifies `12` nats or more and is therefore vacuous on `89.6%` of these questions or worse. |

**We predict PARITY MATTERS, which is the prediction that costs us.** The meter at `k=20` already
reaches `0.618` against selection's `0.190` on this task, and there is no reason a pointwise reward
would help the anchor's draws and not the meter's. We register it because the paper's defence has
never been that the scorer is ours alone --- it is that `log n` buys most of the available utility
for two nats where the meter needs four hundred and eighty --- and an arm that can only confirm us
is not worth the cards.

## Excluded in advance

* Choosing `k` after seeing the result, or reporting the best `k` as if it had been the plan.
* Comparing any level in this pass against a level from another pass (caution (ap)). The committed
  `verifiable_metered_tqa.csv` numbers appear **only** inside G3, as an interval to fall in.
* Presenting the parity arm as a mechanism we propose. Its certificate is `K + log n`, worse than
  either component alone, and the paper must not read as if we recommend it.
* Running the judged `500`-prompt workload. A judged level is position-dominated (caution (m)) and
  this is a question about the mechanism rather than the instrument; the judge-free axis answers it
  without one. **That this arm is one task at one pair is a limitation of the arm and is stated as
  such**, in the same sentence that reports it.
* Repairing a failed gate and re-running. One repair, as everywhere else in this project; a second
  failure closes the question on this corpus.

## Compute

Four arms of `500 x 16` trajectories at `T_max = 24` on `TinyComma-1.8B` + `Llama-3.1-8B-Instruct`,
one card each, plus one `Qwen2.5-7B-Instruct` reward pass per arm. The committed `tqa_metered` ran
`500 x 6` budgets at one trajectory each; this is about `2.7x` that in total and well under the
`24`-GPU-hour escalation threshold per run.

## Scoring log

**Scored 2026-09-21.** All four arms generated and reward-scored on host B cards 4--7.
`analysis/meter_parity.py --report`; CSV `results/meter_parity.csv`.

### Gates

G2 (pipeline) PASSES at all four: every arm records
`(meta-llama/Llama-3.1-8B-Instruct, jacquelinehe/tinycomma-1.8b-llama3-tokenizer)`, the committed
pair.

**G3 FAILS at `k=0.5` (`0.0820` against `[0.0860, 0.1400]`) and at `k=1` (`0.0840` against
`[0.0880, 0.1420]`), so those two arms are INVALID and no band was computed for either.** Both miss
by `0.004`. They PASS at `k=3` (`0.1580` in `[0.1340, 0.2000]`) and at `k=20` (`0.5760` in
`[0.5760, 0.6600]`, at the lower bound exactly).

**What the two failures cost, stated plainly: they were the near-controls.** `k=0.5` and `k=1` were
registered as the budgets where the meter *is* the anchor and the parity arm should therefore land
near selection's own numbers. Losing them means the reading rests on `k=3` and `k=20` with no
control beside it, and that weakens the arm.

**And a spec lesson this registration should have seen.** All four host-B arms read **below** their
local counterparts --- `0.082 < 0.112`, `0.084 < 0.114`, `0.158 < 0.166`, `0.576 < 0.618` --- four
for four in the same direction, with `k=20` landing on its interval's lower bound. G3's reference is
a **local** arm and the new arms are on **host B**, so it is a cross-host comparison, and at the two
budgets where the accuracy is smallest a systematic downward shift of that size exhausts the
interval. That is caution (as)'s regime and caution (at)'s shape: the gate compares two things that
differ in more than the one variable it names. **It is recorded and NOT repaired**, because a gate
rewritten after it fails is no gate (caution (ap)); the honest note is that a host-transfer arm
needs a host-B counterpart as its reference, which did not exist. The registration's single repair
allowance is deliberately left **unused**, so a later session cannot spend it twice.

### B1 --- does the reward model help the meter?

| arm | `acc(1)` | `acc(16)` | paired gain | half-widths | verdict |
|---|---|---|---|---|---|
| metered `k=3` | `0.1580` | `0.1360` | `-0.0220` `[-0.0580, +0.0140]` | `0.61` | **NO EFFECT** (marginal) |
| metered `k=20` | `0.5760` | `0.5880` | `+0.0120` `[-0.0200, +0.0460]` | `0.36` | **NO EFFECT** (marginal) |
| selection over the anchor, **same reward, same call, same pass** | `0.1120` | `0.2100` | **`+0.0980` `[+0.0680, +0.1280]`** | `6.53` | **CLIMBS** |

**This is the answer to the report's objection, and it is a within-pass comparison so caution (ap)
does not touch it.** The report's hypothesis was that selection's gain ``stem[s] largely from the
introduction of the external reward model's alignment signal, an advantage the baseline metered
decoder does not possess''. The metered decoder was given exactly that advantage --- the same
reward model, the same template, the same `score_rewards()` call, the same `16` draws, the same
`500` questions, in the same pass --- and **it gained nothing at either valid budget**, while the
same scorer lifted the anchor's draws by `+0.0980`. The signal is not transferable. What selection
has is not a scorer the meter lacks.

### B2 --- the parity verdict

Best metered `0.594` at `k=20, n=2`, carrying `480.693` nats (`K + log n`, because the budgets
add). Best selection `0.210` at `n=16`, carrying `2.773`. The meter wins on level, and **it wins
only at `k=20`**, a budget the paper's own measurement calls vacuous on `100%` of these questions
(`results/tqa_vacuity_summary.csv`; median `S(x) = 5.86` nats). The registered third branch is more
specific than the first and this registration explicitly anticipated it applying on this grid, so
the reading is **PARITY AT A VACUOUS BUDGET**.

**Both labels are recorded so the softer one cannot look like a choice made after seeing the
number** (caution (ag), and the precedent is feat-136's `REVERSAL REFUTED`): the first version of
the scorer implemented only two of the three registered branches and printed the coarse label,
**PARITY MATTERS**. The point estimate is what it is and the meter does win on level.

**We predicted PARITY MATTERS and the coarse branch was right.**

**A defect in our own specification, recorded and not repaired (caution (w)).** Branch one's
consequence text says the objection is upheld *because* ``a reward model helps the meter too''.
B1 refutes that clause outright. The two halves of that consequence are separable and only the
second is applied: **what selection buys is the budget --- `log n` against `K + log n` --- and not
exclusive access to a scorer.** The first clause is withdrawn rather than quietly kept, and a
future opponent-signal arm should specify B2 in terms of the paired gain, which is within-pass,
rather than a level.

### Post hoc, labelled as such, and it refutes half of our own explanation

Why does the reward not transfer? The obvious story is that a constrained decoder's draws are more
alike, so there is less to choose between. Counting distinct extracted answers among the `16` draws:

| arm | mean distinct of `16` | all `16` identical |
|---|---|---|
| selection over the anchor | `13.26` | `0.4%` of prompts |
| metered `k=3` | `12.83` | `8.6%` |
| metered `k=20` | **`4.87`** | **`22.2%`** |

**True at `k=20` and false at `k=3`.** At `k=20` the decoder is the risky model and its draws
collapse onto one answer on nearly a quarter of questions, so there is genuinely little to select
between. At `k=3` diversity is within `4%` of the anchor's and the reward still buys nothing ---
so there the failure is the *scorer's*, not the decoder's, and it is the same reward
overoptimisation this paper already concedes on TriviaQA. One explanation does not cover both
budgets and we do not pretend it does. Nothing here was registered; it is reported because omitting
a diagnostic that complicates our own story would be one-sided.

### Scope, as registered

One task, one anchor pair, one reward model, two valid budgets, and no judged workload. The two
near-controls are INVALID. The certificate arithmetic (`K + log n`) is a statement about the served
law and is not touched by any of this.
