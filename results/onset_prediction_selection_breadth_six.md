# Pre-registration: is the gain predicted by the anchor's own competence?

Committed **before either arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

Section 6 says *"The strongest anchor gives the largest gain"*, and
`results/selection_breadth.csv` is four anchors. Four points and a sentence is an anecdote. The same
four points carry the paper's explanation of the support ceiling --- that it is a property of the
anchor's overall capability rather than of the domain
(`results/onset_prediction_alpaca_comma7b.md`, A1 CEILING CONFIRMED) --- and that explanation
deserves a test it can fail.

Two anchors turn it into one, and they are chosen so that each is a **controlled** contrast rather
than another point on a scatter:

* **Comma v0.1-1T against Comma v0.1-2T.** Same architecture, same parameter count, same corpus,
  *half the training tokens*. Nothing varies but how much the anchor learned.
* **Pleias-3B against Pleias-1.2B.** Same family and corpus, `2.5\times` the parameters.

Six anchors in three families, with two matched pairs inside them.

## What is run

`h1.py --k-values 0.0`, self-paired (`--safe-model-path` = `--risky-model-path` = the anchor, which
satisfies the factory's shared-vocabulary requirement at `k=0` and draws from the anchor alone),
`--trajectories-per-prompt 8`, the same `500` ordinary prompts (200 neutral, 150 creative, 150
factual), the same temperature and `200`-token cap as every anchor already on record. Then
`analysis/selection_scaling.py` with the same pointwise reward, the same two judges and the same
unconstrained `Llama-3.1-8B-Instruct` opponent, and `analysis/selection_breadth.py` to aggregate.

Nothing else changes. The registered entry gate is unchanged --- mean generated length above `20`
tokens and fewer than `5\%` empty completions on the `n=1` arm --- and a failure is **reported, not
exempted**, as it was when the audited anchor came back at `6.8\%` empty.

## Bands, committed before the run

Read on judge B, the registered scorer; judge C reported beside it and never substituted for it.

**C1 -- do the new anchors gain?** Each anchor's `n=8` gain over its own `n=1`.

| reading | band |
|---|---|
| BOTH GAIN | both new anchors' 95% intervals exclude zero |
| ONE GAINS | exactly one does |
| NEITHER | neither does |

**C2 -- is the gain predicted by the anchor's own competence?** Spearman of the `n=8` gain against
the anchor-alone level `u_{n=1}`, over **all six** anchors, averaging the two judges per anchor as
`selection_breadth.py` already does.

| reading | band |
|---|---|
| TRACKS CAPABILITY | `\rho \ge +0.6` |
| NO TREND | `-0.6 < \rho < +0.6` |
| INVERTS | `\rho \le -0.6` |

**This is the band that can cost the paper a sentence, and the cost is fixed here.** Under NO TREND
or INVERTS, *"The strongest anchor gives the largest gain"* comes out of Section 6, and the
support-ceiling reading of A1 loses its stated mechanism and must be reported as a fact about two
anchors rather than as a property of capability. Under TRACKS CAPABILITY the sentence is replaced by
the coefficient and the number of anchors, which is a stronger and more falsifiable claim than the
superlative it replaces.

**C3 -- the two controlled contrasts.** Within each family, does the more capable sibling gain more?

| reading | band |
|---|---|
| CONTROLLED CONFIRMATION | both contrasts run in the predicted direction (Comma-2T > Comma-1T, Pleias-3B > Pleias-1.2B) |
| SPLIT | one does, one does not |
| CONTRADICTED | neither does |

C3 is the part C2 cannot give: a rank correlation over six heterogeneous models confounds size,
family, corpus and tokenizer at once, and these two contrasts vary one thing each. **A SPLIT or
CONTRADICTED reading is reported in the main text**, because it would say the trend in C2 is carried
by something other than competence.

**C4 -- what may not be claimed.** Leakage is not re-run here and no `B3` number comes from this
arm; each new anchor's near-verbatim recall must be measured by
`analysis/selection_extraction.py` before any leakage statement covers six anchors. Section 6's
"It is `0.0000` at all four anchors" stays at four until then.

## Excluded alternatives

- Dropping a new anchor that fails the entry gate instead of reporting the failure.
- Reading C2 on the judge that gives the larger coefficient, or on one judge alone when the
  aggregation on record averages both.
- Adding a seventh anchor after seeing these two, to move a coefficient.
- Re-reading B1 (`GENERALISES`, already scored at four anchors) on six. B1 is scored and closed;
  this arm adds C1--C3 and does not re-open it.
- Quoting a level from this pass beside a level from another (the cross-pass floor is about `0.04`).

## Scoring log

## Scoring log

### C4 note, 2026-09-14: the first Pleias-3B leakage run was discarded before it was read

The arm ran to `exit=0` at 07:42 and reported `0.0000` at every `n`, which is the registered C4
reading. **It is not used**, and the reason is not the leakage column.

`scripts/run_breadth_leakage.sh` had been given `--batch-size 64` that morning, against the default
`32` that every anchor on record used. The `k=-1` arm is the memoriser sampled alone, and
`generate()` consumes the RNG once per chunk, so the batch size changes the draw: the baseline read
`0.4434 / 85.0%` against `0.3925 / 78.0%` at the other three anchors, and per passage it agreed with
them on **24 of 100** where they agree with each other on **100 of 100**.

Nothing about the leakage number is wrong. What is wrong is the *comparison*: the paper's positive
control --- the memoriser does reproduce these passages, so a `0.0000` is the mechanism and not a
broken probe --- is only a control if it is the same number at every anchor, and
`tests/test_selection_claims.py::test_the_memoriser_baseline_is_identical_at_every_anchor` was
written to catch a bug (the safe model's token ids fed to the memoriser) that moves it by `0.04`.
A batch-size change moves it by `0.05` and is indistinguishable from that bug at the CSV, so
accepting this run would have cost the detector.

Both new anchors' leakage arms were relaunched at the default batch size. The discarded CSVs are
kept out of `results/` at `output/abandoned_batch64/`, and the trap is AGENTS.md caution (u).

### C4 scored, 2026-09-14: both new anchors, both at the default batch size

| anchor | `n=1` | `n=8` | `n=64` | max over 100 | `k=-1` memoriser control |
|---|---|---|---|---|---|
| Comma-7B (1T tokens) | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.3925 / 0.8154 / 78.0%` |
| Pleias-3B | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.3925 / 0.8154 / 78.0%` |

Both re-runs reproduce the memoriser baseline **bit for bit on 100 of 100 passages** against the
three anchors on record, which is caution (u) confirmed from the other side: the `0.4434` that the
batch-64 run produced was the batch size and nothing else. **C4 reads NO LEAK at both**, and with
six of six anchors measured Section 6's sentence becomes "It is `0.0000` at all six anchors".

The *gains* stay at four until the two breadth generations land and C1--C3 are read. The two counts
are deliberately separate claims in this pre-registration: leakage is measured by its own arm, which
needs only the anchor, while a gain needs the generation and the judge.

## Scoring, 2026-09-14 --- C1--C3

Runs: `scripts/run_breadth_queued.sh 0 common-pile/comma-v0.1-1t _comma1t` (`score exit=0 at
10:01`) and the same for `PleIAs/Pleias-3b-Preview` (`score exit=0 at 10:39`), then
`analysis/selection_breadth.py --out results` -> `results/selection_breadth.csv`. Read on judge B,
`Phi-3.5-mini-instruct`, the registered scorer; judge C beside it.

| anchor | $u_{n=1}$ | gain, judge B | gain, judge C | entry gate |
|---|---|---|---|---|
| TinyComma-1.8B (audited) | `0.435` | `+0.054 [+0.013, +0.095]` | `+0.073 [+0.027, +0.120]` | **FAIL** (`6.8%` empty) |
| Pleias-1.2B | `0.411` | `+0.029 [-0.012, +0.068]` | `+0.071 [+0.027, +0.115]` | PASS |
| KL3M-1.7B | `0.310` | `+0.039 [+0.005, +0.076]` | `+0.051 [+0.013, +0.090]` | PASS |
| Comma-7B (2T) | `0.450` | `+0.111 [+0.072, +0.148]` | `+0.155 [+0.106, +0.200]` | PASS |
| **Comma-7B (1T)** | `0.458` | `+0.026 [-0.010, +0.063]` | `+0.113 [+0.069, +0.157]` | **FAIL** (`16.6%` empty) |
| **Pleias-3B** | `0.399` | `+0.047 [+0.003, +0.089]` | `+0.097 [+0.056, +0.140]` | PASS |

### C1 --- ONE GAINS

Pleias-3B's interval excludes zero on the registered scorer, `+0.047 [+0.003, +0.089]`. Comma-1T's
does not, `+0.026 [-0.010, +0.063]`. Exactly one of the two new anchors gains. Both gain on judge C,
which is reported and not substituted.

**Comma-1T fails the entry gate at `16.6%` empty completions** against the registered `5%`, and the
excluded alternatives forbid dropping it, so it is reported with its failure. Restricting to the
`417` prompts whose `n=1` completion is non-empty leaves judge B unchanged at `+0.026 [-0.018,
+0.070]` and lifts judge C to `+0.143 [+0.094, +0.193]`: the empty completions are not what is
holding the registered scorer's reading down.

### C2 --- NO TREND, and it costs the paper a sentence

Spearman of the two-judge mean gain against the anchor-alone level `u_{n=1}`, over **all six**
anchors: **`rho = +0.543`**, inside the registered `(-0.6, +0.6)`.

**The committed consequence applies: *"The strongest anchor gives the largest gain"* comes out of
Section 6.** It was true of the four anchors it was written on and it is not a property of
capability: Comma-1T has the *highest* anchor-alone level of all six (`0.458`) and the second
lowest gain on the registered scorer. The support-ceiling reading of A1 accordingly loses its
stated mechanism and is reported as a fact about the anchors measured rather than as a law about
competence.

This band was written to be able to cost us that sentence and it did. It is worth being precise
about what survives: the *level* of the gain still varies by `4.3x` across anchors, and the largest
gain is still at a `7`B anchor. What does not survive is the claim that the ordering is predicted
by the anchor's own judged quality.

### C3 --- CONTROLLED CONFIRMATION

Both contrasts run in the predicted direction, on the two-judge mean the aggregation uses:

| contrast | what varies | more capable | less capable | difference |
|---|---|---|---|---|
| Comma-7B 2T vs 1T | training tokens, `2x` | `+0.1330` | `+0.0695` | **`+0.0635`** |
| Pleias 3B vs 1.2B | parameters, `2.5x` | `+0.0720` | `+0.0500` | **`+0.0220`** |

**This is what C2 could not give.** A rank correlation over six heterogeneous models confounds
size, family, corpus and tokenizer at once and reads NO TREND; two pairs that vary one thing each
both move the way capability predicts. The honest summary is that capability buys gain *within* a
family and does not order *across* families, and Section 6 says exactly that.

**One caveat, disclosed rather than buried.** The first contrast compares a gate-passing anchor
with a gate-failing one, so it is the weaker of the two. On the non-empty subset it survives with a
smaller margin --- `+0.1351` against `+0.0846`, a difference of `+0.0505` rather than `+0.0635` ---
and the direction does not change. The second contrast involves no gate failure at all.

### C4 --- scored above, 2026-09-14

Both new anchors' leakage arms read `0.0000` at every `n` with the memoriser control reproduced bit
for bit on 100/100 passages, so Section 6's sentence is now "at all six anchors". See the C4 note
above for why the first Pleias-3B arm was discarded unread.

### B1 is not re-opened

`selection_breadth.py` prints a B1 line over the anchors that pass the gate; it is ignored here.
B1 was scored at four anchors as GENERALISES and the excluded alternatives close it. Section 6's
B1 sentence keeps its four anchors and its three numbers.
