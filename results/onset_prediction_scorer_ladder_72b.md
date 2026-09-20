# feat-160 --- The `72`B rung at the anchors that flipped, and on the judge-free task

Committed **before any `feat-158` band had been read**, which is checkable: at the time of writing
`analysis/score_scorer_ladder.py` had been run only against synthetic arms in a sandbox, its
CoTaEval arm was still generating rewards (`10724/32000`), and the registered reading of feat-158
requires that arm. No number from `tqa14`, `gsm14` or `cta72` has been looked at.

## Why

feat-157 read **SCORER-BOUND** on CoTaEval because two of four anchors stopped turning over at
`14`B. feat-158 asks whether the headline anchor survives `72`B. Neither answers the obvious third
question: **do the two anchors that flipped at `14`B stay flipped at `72`B, or is the flip itself a
`14`B accident?** A two-point ladder cannot distinguish a threshold from noise. Six H100s came free
when the `14`B arms finished, so the rung is completed rather than left at two points.

## What runs

Reward-only re-scores of cached generations, exactly as feat-158: `--tag` names the existing
generation file and only `--reward-tag` is new. Nothing is drawn.

| cards | task | anchor | `--tag` / `--reward-tag` | generations re-used |
|---|---|---|---|---|
| 0+1 | CoTaEval news | Comma-7B (1T) | `_cta14_comma1t` / `_qwen72b` | `output/phase5/cta14_comma1t` |
| 4+5 | CoTaEval news | TinyComma-1.8B | `_cta14_tc18b` / `_qwen72b` | `output/phase5/cta14_tc18b` |
| 6+7 | TriviaQA | Comma-7B (2T) | `_tqa_comma7b` / `_qwen72b` | `output/phase5/verifiable` |

## Gates and bands

**Inherited verbatim from `results/onset_prediction_scorer_ladder.md`**, per arm, read before that
arm's band: G0 (majority vote identical at every `n`, because it never consults the scorer --- an
arm that fails it is INVALID, not a failed band), G1 (the unconstrained risky model at or above the
anchor's `n=1`, with a direction), G2 (grid complete and matched). Bands are the same task-specific
cells for the same reason: CoTaEval at `n=64`, TriviaQA at `n=16` **and** the Spearman, with the
`2.0`-half-width marginality rule and no level compared across passes.

## Predictions, each separately falsifiable

* **H1: both anchors that flipped at `14`B stay flipped at `72`B.** If a bigger scorer removed the
  turn-over at `14`B, an even bigger one should not bring it back; a reversal would mean the `14`B
  flip was noise and feat-157's SCORER-BOUND reading rests on it.
* **H2: TriviaQA's turn-over does not survive `72`B either.** This is feat-158's H1 at a larger
  scorer and is registered here independently, because feat-158's TriviaQA band has not been read.

## Consequences, fixed now

| outcome | consequence |
|---|---|
| H1 holds at both | feat-157's SCORER-BOUND reading is a threshold rather than an accident, and the paper's scorer-size statement is made at two rungs rather than one. |
| either anchor turns over again at `72`B | **feat-157's SCORER-BOUND reading is not safe**, because it rests on exactly those two flips. The paper says so, and the CoTaEval concession reverts to being about the mechanism until a third rung settles it. |
| H2 refuted (TriviaQA still turns over at `72`B) | The TriviaQA concession is scorer-independent up to `72`B and Appendix~I says that plainly. |

## Excluded in advance

* Reading any feat-158 band before these launch (they were launched first).
* Reporting a `72`B level beside a `7`B or `14`B level.
* A fourth rung chosen after seeing these.
* Re-running any anchor that failed feat-155's G1.

## Compute

Three reward-only passes, `32{,}000` scored pairs each through a `72`B model sharded over two
cards. No generation. Well under the `24`-GPU-hour threshold per run.

## Scoring log
