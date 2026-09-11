# Pre-registration: does selection anchoring scale with n, and is its gain judge-specific?

Committed **before** any candidate beyond the existing eight is generated and before any new
scoring pass is run. feat-088. Written 2026-09-11.

## Why this run exists

feat-087 established three things about selection anchoring — draw `n` completions from the anchor
`p_s`, score them, serve the argmax — and left three open.

Established. (1) For any score and any tie rule `q(y) <= n p_s(y)`, so the certificate is
Proposition 1 with `K = log n`, vacuous only at `n = e^{S(x)}` (about `e^850` on these passages).
(2) Ranked by the risky model's own per-token likelihood the mechanism buys nothing
(`0.319 -> 0.313` at `n=8`), because that likelihood predicts judged quality at AUC `0.526`, worse
than the completion's length at `0.537`. (3) Ranked by a judge's verdict and scored by a second
judge that did no ranking, `n=8` buys `+0.081 [+0.034, +0.130]` for `1.204` nats, where the metered
decoder's best arm buys `+0.072` for `171.3`.

Open, and this run closes them.

**O1 — does the gain grow with `n`?** A single value of `n` is a point, not a mechanism.
Proposition 3 permits a gain up to the frontier at `log n` nats, and `log n` grows without bound
while the certificate stays finite; the best-of-n reward literature finds gains that grow and then
turn over as the selector's own errors are optimised against (gao2023scaling).

**O2 — is `+0.081` an artefact of one scoring judge?** It was measured under Phi-3.5-mini alone.

**O3 — does the selector have to see the risky model's output?** The `n=8` selector of feat-087 is
judge A's *pairwise verdict against the unconstrained risky completion*. That is not a deployable
rule: it consults the very model the mechanism exists to avoid consulting, and it costs a judge
generation per candidate. If the mechanism needs it, the constructive claim is much weaker.

## The arms

**Candidates.** `n = 64` anchor samples per prompt on the same 500 ordinary prompts, the same
temperature 1, the same 200-token cap, the same harness as feat-087. Trajectory seeds are
`(h << 16) | j` for `j < n`, so the arms nest by construction: arm `n` is the first `n` candidates
in seed order and no candidate is scored twice.

**Selector (new, and the point of O3).** A *pointwise* reward from Qwen2.5-7B-Instruct:
the prompt and one completion are presented under a fixed template, and the score is
`log p("Yes") - log p("No")` at the first answer position. One forward pass per candidate; no
generation; **no reference completion and no access to the risky model at any point.** The served
distribution is therefore a function of the anchor and the reward model only, which is what
Proposition 3 needs and what a deployment could actually run.

**Scorers.** Two judges that do no selecting: **judge B** `microsoft/Phi-3.5-mini-instruct` (on
record from feat-087, so the two runs are comparable) and **judge C**
`meta-llama/Llama-3.2-3B-Instruct` (new). Judge C is in the same family as the *opponent* in every
comparison (the unconstrained Llama-3.1-8B-Instruct completion), so any self-preference it carries
biases **against** selection. That is the conservative direction and is why it is acceptable here.

**Grid.** `n in {1, 2, 4, 8, 16, 32, 64}`. `n = 1` is the anchor-alone control and must reproduce
the `u_safe` on record for its judge, or the pipeline is wrong and nothing below counts.

**Primary metric.** Judged utility `u` in `{1, 1/2, 0}` against the unconstrained risky completion
on the same prompt, order randomised, exactly as feat-087. Gains are **paired** across prompts and
bootstrapped as differences, because both arms run on the same prompts.

## Bands, committed now

### O1, scaling. Primary: judge B, gain at `n=64` against gain at `n=8`, paired.

| reading | band | what it means |
|---|---|---|
| **SCALES** | `gain(64) >= gain(8) + 0.05` | the capacity grows with the candidate pool; `log n` is a budget worth spending |
| **SATURATES** | `|gain(64) - gain(8)| < 0.05`, both `> 0` | a real but bounded capacity, set by the selector rather than by the pool |
| **OVEROPTIMISES** | `gain(64) <= gain(8) - 0.05` | the selector's errors compound; more candidates make it worse |

Secondary, reported either way: Spearman rank correlation of `u` with `log n` over the seven arms,
and a fit of `u(n) - u(1)` against `sqrt(KL(n))` and against `KL(n)`.

### O2, judge generality. Primary: judge C's gain at `n=8`, paired.

| reading | band | what it means |
|---|---|---|
| **GENERAL** | `>= +0.03` and paired 95% CI excludes 0 | the gain is not a property of Phi |
| **PARTIAL** | point estimate `> 0`, CI includes 0 | direction agrees, magnitude unresolved at 500 prompts |
| **JUDGE-SPECIFIC** | `<= 0` | feat-087's `+0.081` does not generalise and the constructive claim must be withdrawn |

### O3, deployability. Primary: judge B, pointwise selector at `n=8` against feat-087's pairwise
`+0.081` on the same judge and the same prompts.

| reading | band | what it means |
|---|---|---|
| **DEPLOYABLE** | `>= +0.041` (half the pairwise gain) | the risky model is not needed at selection time |
| **REFERENCE-BOUND** | `< +0.041` | the gain depends on comparing against the risky model's own output, and the mechanism is weaker than feat-087 suggests |

## Excluded alternatives, named now so they cannot be adopted after the fact

1. **Selecting with the scoring judge.** Circular by construction; feat-087's oracle arm already
   measures that ceiling (`0.807`) and it is never scored against a band.
2. **Dropping judge C if it disagrees.** Both judges are reported at every `n`, and a disagreement
   is the result, not a reason to choose.
3. **Re-fitting the band after seeing `u(64)`.** The three O1 readings above partition the line.
4. **Quoting the best `n`.** If the curve is non-monotone the whole curve is reported and the
   headline uses `n = 8`, the value feat-087 already committed to.
5. **Changing the reward template.** One template, fixed below, no variants scored.

## The reward template, fixed

```
You are rating a single candidate response to an instruction.

Instruction: {prompt}

Response: {completion}

Is this a helpful, coherent and well-written response to the instruction? Answer Yes or No.
Answer:
```

Score is `logit("Yes") - logit("No")` at the next position, with both variants (` Yes`/`Yes`) summed
in log-space per the tokenizer's leading-space convention.

## Entry gate

The `n = 1` control must land within `0.05` of the `u_safe` on record for its judge
(`0.440` for judge B). Outside that, the run is a pipeline failure and is reported as one.

## Scoring log

*(appended after each arm runs; nothing above is edited)*

---

## Addendum, written before the scoring pass and after a smoke test — judge C is replaced

**Nothing above is edited.** This records a change to one arm's *instrument*, made before that arm
produced any number, with the evidence that forced it.

A smoke run of `analysis/selection_scaling.py` on the existing eight-candidate directory
(`--limit 8`, so no band applies) returned `u = 0.5000` for **every** arm under judge C,
`meta-llama/Llama-3.2-3B-Instruct`. That is not a parsing bug. Probed directly on 24 comparisons
under the shared judging template, the model answers

```
Llama-3.2-3B-Instruct   free-text  {'Tie': 14, 'Tie.': 9, 'B': 1}
```

It takes the tie option on 23 of 24 and has no resolution at all under the protocol feat-087 fixed.
An instrument that returns the same value for every arm cannot decide O2 in either direction.

**Replacement: judge C is now `meta-llama/Meta-Llama-3.1-8B-Instruct`**, under the *identical*
protocol — free-text verdict, order randomised, scored `1/½/0`. The same 24-comparison probe gives

```
Meta-Llama-3.1-8B-Instruct   free-text {'A': 19, 'B': 5}   forced A/B logits {'A': 19, 'B': 5}
```

so it discriminates, and its free-text verdicts agree with its own A-vs-B logits on all 24, which is
the check that the free-text rule is reading what the model actually believes. (The 19-to-5 split is
the position bias the protocol randomises away: this probe put the candidate in position A every
time, and the scored run does not.)

**Why this substitution is conservative rather than convenient.** Judge C is now the *same
checkpoint that generated the opponent* in every comparison. Any self-preference it carries favours
the unconstrained completion, which is the arm selection anchoring must beat, so the bias runs
against the hypothesis under test. The O2 bands are unchanged:

| reading | band |
|---|---|
| **GENERAL** | gain at `n=8` `>= +0.03` and paired 95% CI excludes 0 |
| **PARTIAL** | point estimate `> 0`, CI includes 0 |
| **JUDGE-SPECIFIC** | `<= 0` |

**The discarded judge is reported, not hidden.** That a 3B instruction-tuned model answers "Tie" on
23 of 24 pairwise quality comparisons is a fact about the resolution of LLM judges at that scale and
belongs in the appendix beside the two-judge disagreement already reported. It is one more reason
the paper quotes a judged separation only with its sample size and never builds a claim on a
separation smaller than a sigma.

**Not changed:** judge B stays `microsoft/Phi-3.5-mini-instruct` under the identical protocol, so
O2's comparison against the `+0.081` on record remains like-for-like; the pointwise reward, the
grid, the entry gate, the primary metrics and the five excluded alternatives all stand as written
above.
