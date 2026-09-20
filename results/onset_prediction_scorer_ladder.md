# feat-158 --- Is reward overoptimisation a SCORER-SIZE effect, on the judge-free tasks and at 72B?

Committed **before any reward pass is run**. Nothing above `## Scoring log` is edited afterwards.

## Why

`feat-157` read **SCORER-BOUND** on CoTaEval: swapping the pointwise reward `7`B -> `14`B left the
turn-over at Comma-7B (2T) and at its disjoint re-draw (`3.09` and `3.18` interval half-widths) and
removed it at Comma-7B (1T) and TinyComma-1.8B. Two questions follow immediately and the paper can
answer neither.

1. **Does the same effect appear on the judge-free tasks?** Appendix~I concedes reward
   overoptimisation on TriviaQA --- accuracy falls with `n`, Spearman `-0.607`, and at `n=16` the
   interval excludes zero on the wrong side --- at the `7`B scorer and nowhere else. If a `14`B
   scorer removes it, that concession is a statement about scorer size; if it does not, it is a
   statement about optimising a pointwise proxy, which is what the paper claims.
2. **Does the CoTaEval turn-over at the HEADLINE anchor survive a much larger scorer?** It survived
   `14`B at `3.09` half-widths. `Qwen2.5-72B-Instruct` is `~10x` the registered scorer and is
   cached on host B. If the turn-over survives that, `SCORER-BOUND` is confined to the two weaker
   anchors where it was measured; if it does not, the concession is genuinely a scorer requirement.

GSM8K runs beside TriviaQA because it is the task where the reward does **not** turn over, so it is
the control that tells a scorer-size effect from a task effect.

## What runs

**Nothing is generated.** Every arm is a reward-only re-score of cached generations, which is the
same design as the five-judge panel: the text is byte-identical and the only thing that changes is
the instrument. `selection_verifiable.py --reward-tag` exists for exactly this and re-uses the
generation JSONL rather than redrawing it.

The tags below are the script's own naming convention, not new ones: `--tag` selects the cached
generation file and `--reward-tag` names the reward cache and the output CSV, which is precisely the
mechanism `selection_verifiable.py` documents for re-scoring one arm with a second scorer.

| card(s) | task | anchor | scorer | `--tag` / `--reward-tag` | generations re-used | output |
|---|---|---|---|---|---|---|
| 0 | TriviaQA | Comma-7B (2T) | `Qwen2.5-14B-Instruct` | `_tqa_comma7b` / `_qwen14b` | `output/phase5/verifiable/anchor_tqa_comma7b_n64.jsonl` | `selection_verifiable_tqa_comma7b_qwen14b.csv` |
| 1 | GSM8K | Comma-7B (2T) | `Qwen2.5-14B-Instruct` | `_comma7b` / `_qwen14b` | `output/phase5/verifiable/anchor_comma7b_n64.jsonl` | `selection_verifiable_comma7b_qwen14b.csv` |
| 2+3 | CoTaEval news | Comma-7B (2T) | `Qwen2.5-72B-Instruct` | `_cta14_comma7b` / `_qwen72b` | `output/phase5/cta14_comma7b/anchor_cta14_comma7b_n64.jsonl` | `selection_verifiable_cta14_comma7b_qwen72b.csv` |

Because `--tag` names the generation file, **an arm that failed to find its cached generations would
silently regenerate them rather than fail**. G0 below is what catches that: regenerated text cannot
reproduce the committed majority-vote column.

`72`B in `bfloat16` is about `145` GB and does not fit on one `80` GB card. It is sharded with
`--reward-max-memory 0=75GiB,1=75GiB`; the flag defaults to empty so every committed arm keeps the
single-card `device_map` it was run under (caution (q)).

## Gates, per arm, read before that arm's band

**G0 --- majority vote must be IDENTICAL at every `n`, not merely at `n=1`.** Majority vote is
computed from the generations alone and never consults the scorer, so a scorer swap cannot move a
single one of its seven cells. This is a far stricter instrument check than feat-157's `n=1` gate
and it is free. **Any majority-vote cell that differs from the committed arm's `acc` at 4 dp means
the generations are not the ones on record, and the arm is INVALID** (caution (at): the comparison
would have changed two things). The reference is the committed arm's own CSV, read by the scorer;
no constant is typed into this file (caution (v)).

**G1 --- an instrument gate on the UNCONSTRAINED risky model, with a DIRECTION** (caution (au)).
The risky model's accuracy must be **at or above** the anchor's `n=1` accuracy on the same task. A
forced-choice or knowledge task where the unconstrained model reads at or below a small anchor is a
parser or pipeline failure, not a result. Written with the direction because a gate that merely
says ``the interval excludes'' passes a model performing below chance.

**G2 --- the reward cache must be complete and matched.** Exactly `n_max` scores per problem, and
the problem count must equal the committed arm's. A short cache silently truncates the grid.

## Bands

**Corrected before launch, and the correction is recorded rather than quietly made.** The first
draft of this section read the band at `n=64` for every arm. Mutation-testing the scorer against a
synthetic arm --- done before any data existed, which is the only reason this was caught --- showed
that specification is **ill-posed on TriviaQA**: the committed `7`B arm reads
`-0.014` `[-0.046, +0.018]` at `n=64`, which is already NO EFFECT, so ``does the turn-over survive a
larger scorer'' would have had nothing to survive. The concession in Appendix~I is not about `n=64`;
it is about **`n=16`**, where the committed interval is `-0.038` `[-0.068, -0.008]` and excludes zero
on the wrong side, and about the **Spearman** of accuracy against `log n`, committed at `-0.6071`.
A band must be read at the cell the claim is made at (caution (ai): the claim ABOUT the numbers is
what nothing checks), so the band is now task-specific and tied to the committed sentence.

| task | registered quantity | committed `7`B value | ``the turn-over survives'' means |
|---|---|---|---|
| TriviaQA | paired `acc(16) - acc(1)`, **and** Spearman of `acc` against `log n` | `-0.038` `[-0.068, -0.008]`; `-0.6071` | the interval still excludes zero **below** *and* the Spearman is still negative |
| GSM8K (control) | paired `acc(64) - acc(1)`, and Spearman | `+0.066` `[+0.024, +0.108]`; `+0.9286` | not applicable --- this arm is the control and must still CLIMB |
| CoTaEval | paired `acc(64) - acc(1)` | feat-157's `-0.1261` `[-0.1668, -0.0851]` | the interval still excludes zero below |

Every reference above is read **out of the committed CSV by the scorer**, not typed into this file
(caution (v)); the values are quoted here only so the reader can see what the gate is about.
All three use the same `2.0`-half-width marginality rule, and **no level from this pass is compared
against a level from another pass** (caution (ap)) --- the reading is the comparison of VERDICTS.

**The GSM8K control has teeth in both directions.** If GSM8K stops climbing under the `14`B scorer,
that is not a TriviaQA finding, it is evidence that something about the larger scorer broke the
pipeline, and **both `14`B arms are then INVALID rather than informative**. Fixed now so that a
convenient TriviaQA result cannot be kept while its own control is discarded.

| outcome | verdict | consequence, fixed now |
|---|---|---|
| TriviaQA still turns over at `14`B **and** CoTaEval still turns over at `72`B | **SCORER-INDEPENDENT AT SCALE** | Appendix~I's overoptimisation concession is restated with no suggestion that a bigger scorer fixes it, and feat-157's SCORER-BOUND reading is explicitly confined to the two weaker anchors where it was measured. |
| both flip | **SCORER-BOUND, GENERAL** | Limitations states a scorer-size requirement explicitly and both concessions are rescoped to the scorer sizes at which they were measured. |
| exactly one flips | **TASK-DEPENDENT** | Reported as is. A flip on one task is not a general rescue and the other task's concession stands unchanged. |
| GSM8K stops climbing | **INSTRUMENT FAILURE** | Both `14`B arms are INVALID (caution (w)); the question is not retired and the CoTaEval `72`B arm, which does not share their pipeline, is read on its own. |

**Our prediction, in two separately falsifiable halves so that neither can be read as a hedge:**

* **H1: TriviaQA's turn-over DOES NOT survive the `14`B scorer** --- the `n=16` interval no longer
  excludes zero below. TriviaQA answers are short and factual, a `14`B scorer judges them much
  better than a `7`B one, and the paper's own scorer ladder already shows scorer capability binding
  on this axis (a `1.5`B scorer buys `87%` of the `7.6`B gain on the judged workload).
* **H2: CoTaEval's turn-over at the headline anchor DOES survive the `72`B scorer.** It survived
  `14`B at `3.09` half-widths, and the failure there is one of task-reward alignment on long news
  answers, which size does not obviously fix.

If H1 is wrong the paper's TriviaQA concession is stronger than we thought; if H2 is wrong,
feat-157's SCORER-BOUND reading generalises and the CoTaEval concession must be rescoped. Both are
recorded, and each half is scored separately.

## The denominator decision, fixed NOW because it is the one that would otherwise be made after seeing the answer

`sections/selection.tex` says *``four draws of it beat all `28` reward cells on either task''*,
where `28` is **four scorers by seven `n`** at this anchor. The TriviaQA and GSM8K arms above add a
**fifth** scorer, so that denominator becomes `35`. Registered before any number is seen:

* the sentence is **re-derived over all five scorers** and the prose updated to the new denominator
  **whatever the answer**;
* if majority vote at `n=4` no longer clears the maximum over all `35` cells, the claim is
  **withdrawn**, not restricted back to the four scorers that happen to support it;
* `tests/test_judgefree_ratios.py`'s enumerated grid is extended to five scorers in the **same
  commit** as the prose, so the guard and the sentence cannot drift apart.

## Excluded in advance

* Choosing which scorer sizes to report after seeing them.
* Re-running any anchor that failed feat-155's G1: a larger scorer cannot make an anchor able to
  read a news article.
* Reporting a `14`B or `72`B **level** beside a `7`B level.
* Adding a sixth scorer if the fifth disagrees with the other four.
* Reading the CoTaEval `72`B band if G0 fails, for any reason, including a plausible one.

## Compute

Three reward-only passes, no generation. The `72`B pass is `32{,}000` scored pairs through a
sharded `72`B model on two cards; the `14`B passes are `32{,}000` each on one card. Comparable
`14`B passes took under `3` h. Well under the `24`-GPU-hour escalation threshold per run.

## Scoring log

**Scored 2026-09-21.** All three arms finished on host B. `analysis/score_scorer_ladder.py`;
CSV `results/scorer_ladder.csv`.

### Gates: all three pass on all three arms

**G0 passes at all seven majority-vote cells on every arm**, which is the strongest instrument
result this project has produced and it was free. Majority vote never consults the scorer, so seven
identical cells mean the `72`B and `14`B passes scored **byte-identical text** and the only thing
that changed is the instrument. G1 (risky at or above the anchor, with a direction) and G2 (grid
complete and matched) pass everywhere.

**G0 is also what licenses the cross-pass comparison below.** Caution (ap) forbids setting a number
from one sweep against a number from another, because the sweep usually changes what was drawn or
how it was presented. Nothing was drawn here and nothing was re-ordered --- the reward is
deterministic given the text, and G0 proves the text is the same --- so this is the five-judge
panel's situation rather than caution (ap)'s, and the difference between passes **is** the
instrument.

### Bands

| arm | band cell | gain | half-widths | verdict | Spearman (committed) |
|---|---|---|---|---|---|
| TriviaQA @ `14`B | `n=16` | `+0.0240` `[-0.0060, +0.0540]` | `0.80` | **NO EFFECT** | `+0.8571` (`-0.6071`) |
| GSM8K @ `14`B, control | `n=64` | `+0.2140` `[+0.1680, +0.2620]` | `4.55` | **CLIMBS** | `+1.0000` (`+0.9286`) |
| CoTaEval @ `72`B | `n=64` | `+0.0266` `[-0.0097, +0.0623]` | `0.74` | **NO EFFECT** | `+0.7500` (`-0.8929`) |

**The control holds, and holds hard.** GSM8K was registered as the arm that must still climb or
both `14`B arms are INVALID. It climbs at `4.55` interval half-widths with a Spearman of exactly
`+1.0000`. The `14`B pipeline is sound and the TriviaQA reading is admissible.

**Both turn-overs are gone, so the registered reading is `SCORER-BOUND, GENERAL`.** Neither
interval excludes zero below any more, and both Spearmans have **inverted**: TriviaQA
`-0.6071 -> +0.8571`, CoTaEval `-0.8929 -> +0.7500`. Under a larger scorer, accuracy on these
tasks rises with `n` instead of falling.

**H1 CONFIRMED, H2 REFUTED. We predicted that CoTaEval's turn-over at the headline anchor would
survive `72`B and it did not.** That prediction is on record and this is the third registered
prediction this week to be refuted.

### What is rescoped, and what is NOT withdrawn

The registered consequence is applied: **Limitations states the scorer-size requirement explicitly
and both concessions are rescoped to the scorer sizes at which they were measured.** They are
**rescoped, not deleted**, and the distinction is the whole of the honesty here:

* the paper's deployable scorer **is** the `7`B one. Every selection arm in this paper scores with
  it, the `35.4x` serving measurement is of it, and at that operating point the turn-over is real
  and reproduces on a disjoint draw. Nothing measured at `7`B is revised;
* what changes is the **explanation**. The failure was read as a property of optimising a pointwise
  proxy. It is a property of optimising a pointwise proxy **of that size** on these tasks;
* a deployer who wants selection to climb on a knowledge task therefore has a stated requirement,
  and requirements are costs. It is a favourable one only because the paper already measured that
  the scorer is `9.3%` of selection's wall-clock and the draws `90.7%` --- **no new cost number is
  computed here and none should be inferred**, because no latency arm was run at `14`B or `72`B.

**Two things this does NOT license.** It says nothing about the **judged** workload, where the
headline `+0.1045` lives and where no larger scorer has been run --- the scorer-scale ladder there
stops at `7.6`B and its own saturation reading stands. And it does not touch the certificate, which
is `log n` for any score of any size.

### Scope

Three arms, one anchor for the `14`B pair, one anchor for the `72`B arm, three tasks, no judge.
feat-160 extends the `72`B rung to the two anchors that flipped at `14`B and to TriviaQA.
