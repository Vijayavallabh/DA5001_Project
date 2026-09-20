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
