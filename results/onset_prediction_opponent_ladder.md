# Pre-registration: is the judged head-to-head a function of the opponent's STRENGTH?

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists

`onset_prediction_second_opponent.md` (feat-153) swapped the committed opponent
`Llama-3.1-8B-Instruct` for `Qwen2.5-14B-Instruct` and the paired difference `g_sel - g_met` went
from `+0.0645 [+0.0300, +0.0995]` (REVERSAL CONFIRMED) to `-0.0065 [-0.0385, +0.0255]`
(REVERSAL UNRESOLVED). The appendix reports that honestly and reports it as far as it goes: *"the
judged head-to-head between the two mechanisms is opponent-dependent"*, and the paper moved its
weight onto the certificate and the judge-free axis in consequence.

**That sentence is built on two points, and the swap moved two things at once** --- family and size
--- so nothing about it is identified. A reader is entitled to ask whether the paper's headline
comparison is an artefact of one particular opponent, and two points cannot answer that.

Measured now, before anything is run, from the two per-prompt files already on disk: the committed
opponent beats the anchor control on **`0.555`** of prompts and `Qwen2.5-14B-Instruct` on
**`0.850`**. **The "second opponent" was not a perturbation.** It is a `0.295` move on a bounded
scale, and neither the registration nor the appendix says so. A difference of gains has very little
room against an opponent that wins `85%` of the time, so **compression** is a live and untested
explanation for the whole result, and it is a far less damaging one than "the comparison is about
the opponent".

## The axis, and why it may be compared across passes

    opponent strength := 1 - mean(u_anchor_k0)

`u_anchor_k0` is the anchor control's **order-averaged** win rate against that opponent. This is a
judged LEVEL, and caution (ap) forbids comparing a level across passes, so the exemption is argued
rather than assumed:

1. the reference text is `output/sweep_plain`, **byte-identical in every pass**;
2. order averaging judges both presentation orders, so it consumes nothing from the rng --- caution
   (ap)'s own measurement is that it is then a deterministic function of the text under a greedy
   judge, reproducing a committed pass to four decimals where single-order moved `0.066`;
3. the judge, its template, its seed and the prompt set are fixed.

So the number differs between passes **only through the opponent's text**, which is the variable
under study. That is the one level comparison these rules permit, and the reason does not
generalise: a *gain* measured against a different opponent is not comparable, because the gain's
own control is also judged against that opponent.

## What is run

Three more opponents, **within one family so that size is the only thing that moves**:
`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`. With the
two on record this gives five opponents, four of them one family spanning `0.5`B to `14`B.

Each generates one completion per prompt on the same `500` ordinary prompts at the settings the
committed opponent used (temperature `1.0`, `max_new_tokens 200`, chat template, seed `1234`), and
then `analysis/order_averaged_h2h.py --baseline-dir <that run>` judges the **same four committed
arms** against it under judge~B, `--seed 7717`, both presentation orders.

**Only `--baseline-dir` changes.** The selection arm, the metered arm, both controls and the reward
cache are the committed generations byte for byte; nothing is re-drawn and nothing is re-scored.

## The entanglement, disclosed in advance, and which way it cuts

The pointwise scorer is `Qwen2.5-7B-Instruct`, so these three opponents share its family. Judge~B
is `Phi-3.5-mini` and is family-clean with respect to both. If a shared family gives the scorer a
stylistic preference the opponents also express, selection's picks look **more like** the opponent
and the judge is pushed toward a draw --- which depresses `g_sel`, lowers `D3`, and pushes **away
from** REVERSAL CONFIRMED. **The entanglement is conservative for the hypothesis H1 tests**, so a
confirmation is not explained by it. A refutation partly could be, and that is stated here so it
cannot be argued afterwards.

## Bands, committed before the run

### H1 --- primary, and on new data only

If the result is driven by opponent strength, then **every new opponent measuring strength below
the committed opponent's `0.555` must read `REVERSAL CONFIRMED`** --- `D3 > 0` with its 95%
interval excluding zero.

| reading | band |
|---|---|
| **STRENGTH SUPPORTED** | every new opponent with strength `< 0.555` reads REVERSAL CONFIRMED |
| **STRENGTH REFUTED** | at least one opponent with strength `< 0.555` reads UNRESOLVED or REFUTED |
| **NOT TESTED** | no new opponent measures below `0.555` |

A new opponent measuring **above** `0.555` is not a test of this prediction, is excluded from H1,
and is reported on its own line. The threshold `0.555` is the committed opponent's measured
strength and is fixed by data already on disk, not chosen here.

### H2 --- the ceiling at the other end, registered so it cannot be discovered

A difference of gains is compressed at **both** ends. Against an opponent the anchor control
already beats, every arm's win rate runs into the same ceiling and `D3` has no room either.

| reading | band |
|---|---|
| **COMPRESSION AT BOTH ENDS** | an opponent with strength `< 0.30` reads UNRESOLVED while one between `0.30` and `0.555` reads CONFIRMED |

If this fires it is reported **instead of** STRENGTH REFUTED for that opponent, because a floor
artefact is not evidence against the strength account --- and it would say something sharper and
worse: that the committed opponent sits near the only place on the scale where the instrument has
room to show anything at all.

### H3 --- exploratory, and labelled one

Spearman of `D3` against measured strength over all five opponents, with exact two-sided
permutation `p`. **Two of the five points are already seen**, so this is descriptive and is
reported as exploratory. It is not a band and nothing is concluded from it alone.

### H4 --- the manuscript consequence, fixed now

- **STRENGTH SUPPORTED.** The concession in `app:h2hrepeat` becomes a measurement instead of an
  anecdote: the judged difference is present against opponents up to strength `~0.555` and
  compresses against stronger ones, with every opponent's strength printed so a reader can place
  them. The abstract's *"against one fixed opponent"* qualifier **stays** --- a compression
  account explains the disappearance, it does not restore the claim at `0.850`.
- **STRENGTH REFUTED.** The concession stands and hardens: the difference is not a function of
  opponent strength, so the opponent-dependence is unexplained, and that sentence moves from the
  appendix into the main text's limitations.
- **COMPRESSION AT BOTH ENDS.** Reported in the appendix as a property of the *instrument*: the
  judged head-to-head can only resolve a difference in a window of opponent strength, and the
  paper's is measured inside it. This is a reason to prefer the judge-free axis and is written as
  one.
- **NOT TESTED.** The arm is reported as having failed to place a point below `0.555` and no
  consequence is drawn. The models were chosen expecting them to be weaker; if a `0.5`B instruct
  model beats the anchor control more often than `Llama-3.1-8B-Instruct` does, that is itself
  worth a sentence and nothing more.

**We predict STRENGTH SUPPORTED**, because `0.555 -> 0.850` is a large move on a bounded scale and
a difference of gains has to shrink somewhere. We note in advance that predicting it does not make
H2 less likely, and H2 is the reading we would least like.

## Gates, read in this order, before any band

- **G1 --- the corpus and the settings.** Each new opponent run must hold `500` completions over
  the same prompt ids as the committed pass, at temperature `1.0`, `max_new_tokens 200`, chat
  template, seed `1234`. Read off the run, not off this document.
- **G2 --- only the opponent changed.** The `--sel-dir`, `--metered-dir`, `--anchor-dir`,
  `--rewards`, `--judge`, `--n`, `--k` and `--seed` arguments must be identical to the committed
  pass's. An arm that changed anything else is INVALID, not failed (caution (w)).
- **G3 --- an opponent that is degenerate is not a weak opponent.** Fewer than `10%` empty
  completions, and a mean completion length within a factor of `3` of the committed opponent's.
  **Both halves are derived from `output/sweep_plain` by the scorer** (`0.0000` empty, `154.08`
  words) and never typed. A run that fails G3 is excluded from H1 and H2 and reported as excluded:
  its low strength would be a parsing or truncation artefact read as capability, which is caution
  (au)'s shape.

G3's reference is derived by the scorer at run time. The numbers quoted above are what it prints
today and are recorded so that a change in them is visible, not so that they can be typed in.

## What may not be claimed

- No certificate, leakage or `s(x)` number. Proposition~1 does not involve a judge or an opponent.
- No judged **level** quoted across passes, with the single exception argued above, which is a
  level against a byte-identical reference and is used only as the ordering variable.
- Nothing about the judge --- `onset_prediction_judge_panel.md` is that question, and this arm
  holds the judge fixed exactly as that one holds the opponent fixed.
- Nothing about the workload. Every pass here is the same `500` ordinary prompts.
- No pooling of a new opponent with the two on record into a single number.

## Excluded alternatives

- Re-generating the selection arm, the metered arm or either control.
- Changing the judge, the seed, the prompt set, `n`, `k` or the decoding settings.
- Adding an opponent after seeing another opponent's result, or dropping one that has been run.
  The ladder is fixed at these three and is the whole ladder.
- Using `Phi-3.5-mini`, `Qwen2.5-14B`, `Qwen2.5-72B`, `Mixtral-8x7B` or `gemma-2-27b-it` as an
  opponent: all five are judges on this paper's panel, and an opponent that is also an instrument
  is not a clean point.

## Scoring log

### Amendment, 2026-09-22 15:40, BEFORE any of the three opponents produced data

`blocklist_decode.py --split ordinary` writes **`850`** generations, while
`order_averaged_h2h.py` judges the **`500`** prompts present in all four arms as well. The first
version of the scorer computed G3 over the whole run, which is a gate on `350` texts no judge was
ever shown. Both gates are now scoped to **the judged intersection**, read off each pass's own
per-prompt file, and a second gate is added:

- **G1 is now explicit and mechanical.** The judged prompt id set must equal the committed pass's,
  as sets. An opponent run that silently lost prompts is not a weaker opponent, it is a smaller
  comparison, and G3 cannot see it --- a `400`-prompt subset has a perfectly ordinary empty rate
  and mean length. An arm failing G1 is excluded from H1 and H2.

The derived reference is **unchanged** by the scoping (`500` judged prompts, `0.0000` empty,
`154.08` mean words), which is the check that this was a correction and not a move.

`tests/test_opponent_ladder.py` gains `test_g1_excludes_an_arm_judged_on_a_different_prompt_set`.
Eleven band and gate mutations now pass, and all of them were written and run **before the three
opponents existed**: the `0.5`B arm was still generating and neither the `1.5`B nor the `3`B
checkpoint had reached host B.

### Incident, 2026-09-22 15:36 --- a model directory is not a model

Two of the three arms died immediately on `LocalEntryNotFoundError`. `hf_cache/` held
`models--Qwen--Qwen2.5-1.5B-Instruct` and `models--Qwen--Qwen2.5-3B-Instruct` on host B as **`24`K
empty skeletons** --- `blobs`, `snapshots`, `trees` and an **empty `refs/`** --- which `ls` renders
identically to the `28`G `Qwen2.5-14B-Instruct` beside them. The model list for this arm was chosen
off that listing. This is caution (q) again (`refs/main` missing, so `HF_HUB_OFFLINE=1` cannot
resolve a cache whose files are on disk) crossed with caution (aw) (a dependency you cannot see is
the one that stops a launch), and the new part is that **the directory existed and was empty**, so
neither of those checks would have caught it.

`scripts/run_opponent.sh` now preflights the way `HF_HUB_OFFLINE` itself resolves --- a non-empty
`refs/main` **and** a snapshot holding `.safetensors` --- and exits `3` before taking a card.

**The more dangerous half of the same incident: the judge ran anyway.** The launcher's steps were a
plain sequence, so a generation that failed was followed by a judging pass against an empty
opponent directory. It was caught only by an assertion inside `order_averaged_h2h.py` about there
being no shared prompt --- an accident of the failure being total. **A *partial* generation would
have judged a quietly smaller prompt set and produced a healthy-looking CSV**, which is exactly what
G1 was added above to catch, and the launcher now aborts on a non-zero generation exit.

### Amendment, 2026-09-22 20:50 --- the committed opponent came from a DIFFERENT GENERATOR

**Written with one of the three arms already read** (`Qwen2.5-0.5B`, which measured strength
`0.7035` and so fell outside H1 as the NOT TESTED branch anticipated) and before the other two
existed. **No band is changed and no threshold moves**; what is added is a column and a caveat,
and it is recorded here rather than discovered later.

Caution (at) says two arms compared must have come from the same pipeline and that the runs record
enough to check it. Checked: they did not.

| opponent | generator |
|---|---|
| `Llama-3.1-8B-Instruct` (committed) | **`h1.py`** |
| `Qwen2.5-0.5B` / `1.5B` / `3B` / `14B` | **`analysis/blocklist_decode.py`** |

`blocklist_decode.py` generates one prompt at a time under a per-prompt seed; `h1.py` batches and
seeds per trajectory index. It is read off the records themselves --- `blocklist_decode` stamps
`blocklist_ngram` into every one and `h1.py` never does --- so `analysis/opponent_strength.py` now
carries a `generator` column and no directory name is trusted.

**What this costs, stated plainly.** H1's threshold is the committed opponent's own strength,
`0.555`, so **H1 is a cross-pipeline reading** and part of any gap between `0.555` and a ladder
point could be the generator rather than the opponent. That does not invalidate it --- the
quantity is the anchor control's win rate against a *text*, and a text is a text however it was
produced --- but it is a confound the reading has to carry, and it was not in the registration.

**What is clean and was the ladder's stated purpose.** The four Qwen opponents share one
generator, one family and one prompt set, with size the only variable, which is exactly what the
registration says the ladder is for (*"within one family so that size is the only thing that
moves"*). The within-pipeline four-point ordering is therefore the arm's primary exploratory
content, and the committed Llama point is reported beside it as the paper's existing reference
rather than as a fifth rung.

The `blocklist_decode` route is not new here: `onset_prediction_second_opponent.md` (feat-153) used
it for `Qwen2.5-14B`, and that arm's `-0.0065 [-0.0385, +0.0255]` is the concession in
`app:h2hrepeat`. So the confound is already in the paper, undisclosed, and this ladder is what
surfaced it. The appendix sentence will say which generator produced which side.

## Scoring, 2026-09-22 --- gates, then H1, then H2, then H3

### Gates

| gate | requirement | measured | reading |
|---|---|---|---|
| G1 | the judged prompt id set equals the committed pass's | `500`/`500` on all three | **PASS** |
| G3 | `<10%` empty, mean length within `3x` of the committed opponent's `154.08` words | `0.0%` empty at every arm; `123.5`, `114.6`, `118.9` words | **PASS** |
| generator | (added by amendment, not registered) | committed `h1.py`, all four ladder arms `blocklist_decode` | **MIXED** |

No arm is excluded. The mixed generator is a caveat on H1's threshold, recorded above.

### The ladder

| opponent | strength | paired `D3` | verdict | generator |
|---|---|---|---|---|
| `Llama-3.1-8B-Instruct` (committed) | `0.555` | `+0.0645 [+0.0300, +0.0995]` | **CONFIRMED** | `h1.py` |
| `Qwen2.5-0.5B-Instruct` | `0.704` | `+0.0490 [+0.0135, +0.0850]` | **CONFIRMED** | `blocklist_decode` |
| `Qwen2.5-1.5B-Instruct` | `0.773` | `+0.0195 [-0.0165, +0.0555]` | UNRESOLVED | `blocklist_decode` |
| `Qwen2.5-3B-Instruct` | `0.825` | `-0.0200 [-0.0540, +0.0140]` | UNRESOLVED | `blocklist_decode` |
| `Qwen2.5-14B-Instruct` (on record) | `0.850` | `-0.0065 [-0.0385, +0.0255]` | UNRESOLVED | `blocklist_decode` |

### H1 --- **NOT TESTED**

**No new opponent measured below the committed one's `0.555`.** The band required every opponent
weaker than that to read REVERSAL CONFIRMED; none was weaker, so the primary hypothesis is
untested and is reported as untested. That is the fourth branch of the registration, and it says
what to do: *"if a `0.5`B instruct model beats the anchor control more often than
`Llama-3.1-8B-Instruct` does, that is itself worth a sentence and nothing more."*

**It is worth the sentence.** `Qwen2.5-0.5B-Instruct` --- sixteen times smaller than the committed
opponent --- beats the anchor control on `70.4%` of prompts against the `8`B Llama's `55.5%`, and
it is **not length**: the Llama's completions are the LONGEST of the five (`154.1` words against
`114.6`--`133.3`), and length usually helps under an LLM judge. **Opponent strength on this axis is
not parameter count.** Within the Qwen family it is ordered by size (`0.704`, `0.773`, `0.825`,
`0.850` at `0.5`, `1.5`, `3`, `14`B), so size orders strength *within* a family and the family
shift dominates across them.

**We predicted STRENGTH SUPPORTED and did not get to test it.** The prediction is neither
confirmed nor refuted and is not quietly converted into the exploratory result below.

### H2 --- **NOT FIRED**

No opponent measured below `0.30`, so the floor branch had nothing to fire on. The registered
concern --- that the committed opponent sits near the only place the instrument has room --- is
untested at the low end and remains open.

### H3 --- exploratory, and labelled

Two series, because the generators are not uniform:

| series | Spearman | exact two-sided `p` |
|---|---|---|
| all five, **mixed** generators | `-0.900` | `0.0833` |
| the four Qwen arms: **one generator, one family, size the only variable** | `-0.800` | `0.3333` |

`p = 0.333` is the *smallest* value four points can produce short of a perfect ordering, so the
four-point series cannot be significant and is not offered as significance. What it shows is the
**shape**: `+0.0490`, `+0.0195`, `-0.0200`, `-0.0065` against strengths `0.704`, `0.773`, `0.825`,
`0.850`. The difference decays as the opponent strengthens, crosses zero between `0.773` and
`0.825`, and the two points past the crossing are both UNRESOLVED rather than reversed.

Two points already seen enter both series. Neither is a band.

### What this does to the manuscript

`app:h2hrepeat` says the judged difference *"does not survive"* a second opponent, on two points
whose swap moved family and size together. It now has **five points on a measured axis**, and the
sentence becomes specific rather than anecdotal:

- the second opponent was **not a perturbation** --- it moved the opponent's win rate against a
  fixed reference from `0.555` to `0.850`, a `0.295` move on a bounded scale;
- the difference is **CONFIRMED at `0.555` and `0.704`** and **UNRESOLVED at `0.773` and above**,
  so the boundary lies between them;
- **the strength account is consistent with all five points and is not established by them**:
  H1 was the test and it went untested, the four-point series cannot reach significance, and the
  five-point series mixes generators. It is offered as the shape the data have, not as a mechanism.

The abstract's *"against one fixed opponent"* qualifier **stays**. A compression account explains
the disappearance at `0.850`; it does not restore the claim there.

**The generator disclosure goes into the appendix**, as the amendment promised: the committed
opponent's completions come from `h1.py` and every alternative opponent this paper has ever used
--- including the `Qwen2.5-14B` arm whose `-0.0065` is the published concession --- comes from
`analysis/blocklist_decode.py`.

### Commands

```
bash scripts/run_opponent.sh Qwen/Qwen2.5-0.5B-Instruct qwen05b 1
bash scripts/run_opponent.sh Qwen/Qwen2.5-1.5B-Instruct qwen15b 4
bash scripts/run_opponent.sh Qwen/Qwen2.5-3B-Instruct   qwen3b  5
.venv/bin/python analysis/opponent_strength.py --out results
```

`results/opponent_ladder.csv`, `results/order_averaged_h2h__opp_qwen{05b,15b,3b}.csv`.

### Post-hoc, launched 2026-09-22 21:35 --- judge C on three rungs, and which way it cuts

**Registered instrument: judge~B only.** The registration says a second judge here *"would be
post-hoc and would be labelled so"*, which is what this is. It is launched **after** the ladder was
scored and it cannot change H1, which is NOT TESTED and stays NOT TESTED. The only question it can
answer is whether the decay's shape is one instrument's.

**Stated before the numbers exist: judge~C cuts toward the result, not against it.** Judge~C is
`Meta-Llama-3.1-8B-Instruct`, which is the **metered arm's own risky model** --- the
self-preference relation this paper already discloses (caution (aa)). If it favours anything it
favours the metered arm, which pushes `g_sel - g_met` DOWN and makes the decay look stronger. So a
confirmation under judge~C is weak evidence and a *refutation* would be strong. Whichever way it
reads, it is reported with this sentence attached.

Three rungs: `qwen05b`, `qwen15b`, `qwen3b`. The committed opponent's judge~C reading is already on
record and the `14`B arm's is not re-run.

```
bash scripts/run_opponent_judge.sh qwen15b meta-llama/Meta-Llama-3.1-8B-Instruct judgeC 1
```

### Scoring the post-hoc judge, 2026-09-22 --- **THE DECAY IS NOT REPRODUCED**

| opponent | strength | judge~B | judge~C | shift | judge~C reading |
|---|---|---|---|---|---|
| `Qwen2.5-0.5B` | `0.704` | `+0.0490 [+0.0135, +0.0850]` | `-0.0090 [-0.0555, +0.0375]` | `-0.0580` | UNRESOLVED |
| `Qwen2.5-1.5B` | `0.773` | `+0.0195 [-0.0165, +0.0555]` | `-0.0015 [-0.0455, +0.0425]` | `-0.0210` | UNRESOLVED |
| `Qwen2.5-3B`   | `0.825` | `-0.0200 [-0.0540, +0.0140]` | `-0.0240 [-0.0555, +0.0085]` | `-0.0040` | UNRESOLVED |

Over these three rungs judge~B's rank correlation with strength is **`-1.000`** and judge~C's is
**`-0.500`** at exact `p = 1.0000` --- no ordering at all. Judge~C reads **`0` of `3`** rungs
CONFIRMED against judge~B's `1`.

**This is the refutation branch, and it was named as the strong one before the numbers existed.**
The decay in H3 is a property of judge~B on this ladder and is not reproduced by a second
instrument. H1 is untouched --- it was NOT TESTED and stays NOT TESTED.

**And the direction argument written in advance was half right, which is worth recording.** It said
judge~C favours the metered arm and so pushes `D3` down, making the decay look *stronger*. Judge~C
did push every rung down --- but **not uniformly**: `-0.0580`, `-0.0210`, `-0.0040`, largest where
the opponent is weakest. A strength-dependent shift **flattens** a decaying series rather than
steepening it, and the pre-stated argument was about the LEVEL and did not anticipate that. The
prediction is therefore not scored as confirmed; what was right is that a refutation under this
judge is strong evidence, and that is the branch that fired.

The mechanism is at least coherent: against a weak opponent every arm wins often, so a judge with
a preference for the metered arm's own checkpoint has the most room to express it exactly where
judge~B saw the largest positive difference.

**Manuscript consequence.** The appendix paragraph written from the judge~B ladder is corrected in
place: the five-point series stays, because it is what judge~B measures and the concession it
sharpens is real, but the sentence now says the decay is **one judge's** and that a second judge
reads every rung unresolved with no ordering. The strength account is **weaker** after this arm
than before it, and the paper says so.

```
bash scripts/run_opponent_judge.sh qwen05b meta-llama/Meta-Llama-3.1-8B-Instruct judgeC 5
```

`results/opponent_ladder_judgeC.csv`, `results/order_averaged_h2h__opp_qwen{05b,15b,3b}_judgeC.csv`.
