# Pre-registration: scorer family by judge family, and the other scorer at a matched certificate (feat-212)

**feat-212.** Committed **2026-09-25**, before any judge call of the passes below. Nothing above
`## Scoring log` is edited after the first verdict.

## Why

Review 2 (Q4): "the headline utility claim rests on one scorer family plus a small judge ... provide a
scorer-family x judge-family factorial"; its W3: the largest reading comes from a judge in the scorer's family.
Review 1's roadmap: a non-Qwen scorer in the primary comparison. feat-198 scored the headline pool with
`gemma-2-27b-it` and judged it under judge~B only (`+0.015 [-0.0185, +0.0485]`, UNRESOLVED). This judges the
same two selections, one per scorer, under every judge of the panel, and puts the gemma-scored selection into
the matched-certificate comparison.

## What runs (host B, GPUs 4-7; `scripts/run_feat211.sh`, which queues both features)

`analysis/matched_h2h.py`, both presentation orders, de-echoed text, `1,200` characters, the committed opponent,
each judge on host B, where the committed panel was judged. Arms: `sel_n64` (`sel:64`, the Qwen2.5-7B reward,
`results/selection_rewards64.csv`), `sel_g64` (`sel:64:results/selection_rewards64_gemma27b.csv`, the same `64`
candidates under the gemma reward), `sel_n1`, `metered_k10`, `anchor_k0`, with their committed specifications;
controls `sel_*` against `sel_n1`, `metered_k10` against `anchor_k0`.

- judges C (`Meta-Llama-3.1-8B-Instruct`), D (`Qwen2.5-72B-Instruct`, two cards), E (`Mixtral-8x7B-Instruct-v0.1`,
  two cards), F (`Qwen2.5-14B-Instruct`), fresh caches, tags `fact_C`, `fact_D`, `fact_E`, `fact_F`;
- judge~B and judge~G: the two passes feat-211 registers (`cpk_B_hostb`, `cpk_G`) carry these five arms, plus
  `win_4.16` and `frontpw_4.16` with control `win_0`.

**Differences, per judge.** `S = sel_n64 - sel_g64` (the scorer effect; the two share the control `sel_n1`, so
this is the level difference, paired); `D3_q = sel_n64 - metered_k10` and `D3_g = sel_g64 - metered_k10`.
Matched certificate (judges B and G): `sel_g64 - win_4.16`, `sel_g64 - frontpw_4.16`.

## Predictions

- **F1** `S` is **CONFIRMED** (the Qwen-scored selection is judged better) under at least **five of the six**
  judges B, C, D, E, F, G.
- **M1** at the matched `log 64`, the gemma-scored selection minus the windowed meter at `W = 4.16` and minus the
  front-loaded pathwise meter: **CONFIRMED** under judge~B and under judge~G.

**Descriptive, no band.** The 2 x 6 table of `D3` (scorer by judge); `S` under the two Qwen-family judges (D,
F) set beside `S` under judge~G, the one judge in the gemma scorer's family (the circularity reading); each
fresh pass's reproduction of the committed panel's per-prompt levels for the four committed arms under the same
judge; empties served by each scorer and how each judge scores them.

## What the manuscript does with each outcome, fixed now

The 2 x 6 table goes to the judge-validity appendix; Section 4's scorer sentence reports the count of judges
under which the gemma-scored headline resolves, whatever it is. If F1 holds, the paper says the Qwen reward
picks better drafts under every judge family it was read by, and circularity would have to reverse that; if it
fails under a judge in the gemma scorer's family and holds under the Qwen-family judges, the paper says the
reading tracks the judge's family. M1 is reported in the matched-certificate paragraph of the appendix.

## Excluded in advance

- Any judge, scorer, opponent, template or truncation chosen after a verdict is read; pooling across passes;
  comparing a level across judges.

## Scoring log

### Scored 2026-09-25 (host B, six judge passes) --- M1 right, F1 wrong

Command: `.venv/bin/python analysis/score_feat211.py` -> `results/scorer_judge_factorial.csv` (the 2 x 6 table)
and the feat-212 rows of `results/cpk_baseline_scoring.csv`. Judges B and G are the two feat-211 passes
(`cpk_B_hostb`, `cpk_G`), and C-F are `fact_C` .. `fact_F`, all on host B.

| judge | `S` = Qwen-scored minus gemma-scored | `D3`, Qwen-scored | `D3`, gemma-scored |
|---|---|---|---|
| B `Phi-3.5-mini` | `+0.044` `[+0.023, +0.065]` CONFIRMED | `+0.059` CONFIRMED | `+0.015` `[-0.0185, +0.049]` unresolved |
| C `Llama-3.1-8B` | `+0.0175` `[-0.0095, +0.0455]` | `+0.0765` CONFIRMED | `+0.059` `[+0.010, +0.109]` CONFIRMED |
| D `Qwen2.5-72B` | `-0.0215` `[-0.0455, +0.002]` | `+0.055` CONFIRMED | `+0.0765` `[+0.0285, +0.126]` CONFIRMED |
| E `Mixtral-8x7B` | `+0.010` `[-0.0185, +0.0375]` | `-0.0015` unresolved | `-0.0115` `[-0.056, +0.033]` unresolved |
| F `Qwen2.5-14B` | `-0.0115` `[-0.0345, +0.0115]` | `+0.121` CONFIRMED | `+0.1325` `[+0.089, +0.1765]` CONFIRMED |
| G `gemma-2-27b-it` | `-0.040` `[-0.067, -0.0135]` REFUTED | `+0.0545` CONFIRMED | `+0.0945` `[+0.039, +0.148]` CONFIRMED |

- **F1 WRONG.** `S` is confirmed under **one** judge, B, the pre-specified one. It is unresolved under C,
  D, E and F, and refuted under G, the gemma scorer's own family.
- **M1 RIGHT.** At the matched `log 64` the gemma-scored selection beats the windowed meter by `+0.054`
  `[+0.0215, +0.086]` and the budget-up-front meter by `+0.0455` `[+0.014, +0.0775]` under B. Under G the
  margins are `+0.263` and `+0.2425`, all four confirmed.

**Descriptive.**
- The gemma-scored headline (`D3_g`) resolves under **four of six** judges (C, D, F, G), against five of six
  for the Qwen-scored one. The judge-B reading that made the headline "depend on the scorer" is the one
  judge under which the gemma-scored headline does not resolve (with E, which reads nothing).
- The circularity reading: the two Qwen-family judges do not prefer the Qwen-scored drafts (`S = -0.0215`
  and `-0.0115`, both unresolved), while the gemma-family judge prefers the gemma-scored ones (`-0.040`,
  refuted). So there is a same-family preference in the gemma family and none measurable in the Qwen family.
  F's `+0.121`, the largest headline reading, is not the Qwen scorer's doing: F reads the gemma-scored
  headline at `+0.1325`.
- Reproduction: on the same host each fresh pass gives the committed de-echoed panel's `D3` exactly (C
  `0.0765`, D `0.055`, E `-0.0015`, F `0.121`, G `0.0545`). Judge~B, committed on a local A100, reads
  `0.059` here against `0.0505` there (caution (as)).

**What the manuscript does (as fixed above).**
- Section 4's scorer sentence reports that the gemma-scored headline resolves under four of six judges.
- The judge-validity appendix carries the 2 x 6 table.
- F1's failure is stated: only judge~B prefers the Qwen reward's drafts.
- The registered rule for a failure under the gemma-family judge ("the reading tracks the judge's family")
  applies only in part, because F1 also fails under the Qwen-family judges. The paper says what was
  measured: a same-family preference under G, none under D or F.
