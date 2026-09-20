# feat-157 --- Is CoTaEval's turn-over a property of the scorer, or of the approach?

Committed **before any generation**. Nothing above `## Scoring log` is edited afterwards.

## Why

`feat-155`/`feat-156` found that on CoTaEval news the registered pointwise reward
(`Qwen2.5-7B-Instruct`) does not merely fail to help --- it **turns over at all four anchors that
can read the articles**, costing the headline anchor `43%` of its own F1 by `n=64`
(`-0.1836 [-0.2249, -0.1412]`, reproduced on a disjoint draw at `-0.2334`).

A deployer reading that has one immediate question, and the paper cannot currently answer it: **is
this the mechanism, or is it the 7B scorer?** If a stronger scorer removes the turn-over, the
finding is a statement about scorer capability and the mechanism survives with a stated
requirement. If it does not, the finding is about selection under a pointwise reward on this
workload, which is a far heavier limitation and belongs in Limitations as such.

This is the same axis `analysis/serving_latency.py` already settled for **cost** --- where the
parameter-count story was wrong and the measurement showed the draws, not the scorer, dominate
(caution (ae)). It has never been settled for **utility**.

## What runs

Identical to feat-155 in every respect --- same `500` CoTaEval news items, same anchors, same
`n in {1..64}`, same `--batch-size 16`, same `--seed 8801`, same SQuAD token-F1 --- changing
**only the reward model**, from `Qwen2.5-7B-Instruct` to **`Qwen2.5-14B-Instruct`**, at the four
anchors that passed G1:

| card | anchor | scorer | tag |
|---|---|---|---|
| 0 | Comma-7B (2T) | `Qwen2.5-14B-Instruct` | `_cta14_comma7b` |
| 1 | Comma-7B (1T) | " | `_cta14_comma1t` |
| 2 | TinyComma-1.8B | " | `_cta14_tc18b` |
| 3 | Comma-7B (2T), seed `5254` | " | `_cta14_s5254` |

The four anchors that failed G1 are **not** re-run: a stronger scorer cannot make an anchor able to
read a news article, and running them would only re-measure the capability floor.

## Gates

G1, G2 and G4 exactly as feat-155, per anchor, read before that anchor's band. An anchor's `n=1`
F1 must be unchanged from feat-155 to within bootstrap noise, since `n=1` does not involve the
scorer at all --- **if it moves, the pipeline changed and the arm is INVALID**, which is the
cleanest instrument check available here and is fixed now.

## Bands

Per anchor, paired `F1(64) - F1(1)` within this pass, with the same CLIMBS / NO EFFECT / TURNS OVER
table and the same `2.0`-half-width marginality rule.

**The primary reading is the comparison of verdicts, not of levels** (no level is compared across
passes, caution (ap)):

| outcome | verdict | consequence, fixed now |
|---|---|---|
| all four still TURN OVER | **SCORER-INDEPENDENT** | The limitation is about selection under a pointwise reward on this workload, not about scorer size. Limitations says so, and the appendix drops any suggestion that a bigger scorer would fix it. |
| the turn-over disappears at two or more anchors | **SCORER-BOUND** | The finding is rescoped to the `7`B scorer, the paper states the scorer requirement explicitly, and the headline is qualified rather than withdrawn. |
| mixed (one anchor flips) | **UNRESOLVED** | Reported as is; a single flip at a marginal anchor is not a rescue, and `TinyComma-1.8B` was already the marginal one at `1.84` half-widths. |

**We predict SCORER-INDEPENDENT.** Reward overoptimisation is a property of optimising against a
proxy, and a larger proxy is still a proxy; the paper's own TriviaQA result showed the same shape at
this scorer size. Registering the prediction means a SCORER-BOUND result counts against us.

## Excluded in advance

* Re-running the four anchors that failed G1.
* Reporting a `14`B level beside a `7`B level.
* Treating a single anchor's flip as a rescue.
* Choosing a third scorer after seeing these results and reporting whichever agrees.

## Compute

Four arms of `500 x 64` at `7`B anchor plus a `14`B scorer, one card each on idle H100s;
comparable arms took under `3` h. Well under the `24`-gpu-hour threshold per run.

## Scoring log
