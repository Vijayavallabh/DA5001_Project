# Pre-registration: the headline judged on whole responses (feat-204)

**feat-204.** Committed **2026-09-25**, before the judge is called. Nothing above `## Scoring log` is
edited after the first judge call.

## Why

Review 1 (sixth round, Q4) asks how much of the headline survives "full-response judging". Every judge
pass on record shows the judge the first `1,200` characters of the prompt and of each response
(`analysis/utility.judge_batch`). Measured on the headline's texts before this registration: `19` of the
`500` opponent texts (`3.8%`), `20` of the meter's at `k=10` (`4.0%`), `2` of selection's (`0.4%`) and none
of the anchor's or of any prompt exceed `1,200` characters, so the cut touches about one pair in
twenty-five and falls mostly on the two arms whose texts are longer.

## What runs

`analysis/order_averaged_h2h.py --deecho --judge-max-chars 0 --tag fullresp --out results`: the
headline pass with nothing cut (the tokenizer limit is raised to `8,192` and the script asserts no item
reaches it), judge B, seed `7717`, both orders, every other input the headline's. Host B, one card.
Its reference is the same pass on the same host with the default cut, which feat-201's judge passes
already computed (`results/order_averaged_h2h_blockwise_blk200n1.csv`, D1 to D3).

## Readings and predictions

- **R1.** `D3` of the uncut pass reads **CONFIRMED**, as the cut pass on the same host does
  (`+0.059 [+0.0235, +0.094]`). *Predicted: CONFIRMED.*
- **R2.** The uncut `D3` lies within `0.02` of the cut one. *Predicted: within.*
- **Descriptive.** `D1`, `D2`, and per class (biographies separately, as the review asks), the paired
  difference of each arm's level between the uncut and the cut pass on the prompts where a text was
  cut.

## What the manuscript does with each outcome, fixed now

The judging paragraph of the appendix states the cut and its incidence, and reports R1 and R2. If R1
fails, the headline sentence in Section 4 carries the uncut reading beside the cut one.

## Excluded in advance

- Any other character limit, judge or seed chosen after a verdict is read.

## Scoring log

### Scored 2026-09-25 09:22 IST --- R1 CONFIRMED, R2 WITHIN: nothing the cut removed moves the headline

The pass ran on host B, GPU 5, `09:08` to `09:10` IST (`output/logs/feat204_judge.done`, launcher
`scripts/run_feat202_206.sh q5`); the script's own assertion that no uncut item reaches `8,192` tokens held.
Scored by `.venv/bin/python analysis/score_feat203_206.py --only 204` -> `results/full_response.csv`.

| reading | value | registered | verdict |
|---|---|---|---|
| R1, `D3` with nothing cut | `+0.062 [+0.0265, +0.0975]`, CONFIRMED | CONFIRMED | **right** |
| R2, gap to the cut pass on the same host (`+0.059 [+0.0235, +0.094]`) | `0.003` | within `0.02` | **right** |

**Descriptive, no band.** `D1 = +0.113 [+0.088, +0.138]` and `D2 = +0.051 [+0.0255, +0.076]` uncut. On the
`150` biographies alone the uncut difference is `+0.0217 [-0.0517, +0.0933]`. Any arm's level moved on `23`
of `500` prompts when nothing was cut.

**Manuscript, as registered.** The judging paragraph of the appendix states the cut, its incidence and
R1-R2; the headline sentence is unchanged because R1 holds.
