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
