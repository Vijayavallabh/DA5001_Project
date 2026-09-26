# feat-217: selection with a vetted scorer, prompt-blind and licensed (review 8, W5 and Q2)

Registered 2026-09-26 17:05 before any score was computed. Nothing above `## Scoring log` is edited afterwards.

**Question.** The paper's vetting condition asks for a scorer that "neither holds the work nor takes the user's
instructions"; the headline scorer (Qwen2.5-7B-Instruct, web-trained, reads the prompt) meets neither, so its window
certificate holds for 38 queries in total. What does selection keep with a scorer that meets both?

**Scorer.** Comma-7B (`common-pile/comma-v0.1-2t`, trained on openly licensed text, the largest anchor that passes the
paper's vetting screen), scoring each of the committed pool's 64 draws per prompt (`output/phase5/sel_anchor64`, the
de-echoed served text) by its mean per-token log-likelihood **with no prompt in context**. An empty draw scores
`-1e9`, so it is never served while a non-empty one exists (the non-empty rule, still certified at `log 64`).
`analysis/promptblind_scorer.py`, sharded over host B's H100s -> `results/selection_rewards64_promptblind.csv`, the
same format as `results/selection_rewards64.csv`.

**Arm.** `sel_pb` = the argmax of that score over the 64 draws (`analysis/matched_h2h.py` `sel:64:<cache>`), judged
against the committed opponent in both orders; control `sel_n1`. Judge B on the local A100s with the committed
comparators' verdicts carried over (as feat-216); judge G on host B's H100s with feat-216's H100 comparator verdicts.

**Quantities, per judge.** `gain(sel_pb)` over `sel_n1`; `DQ = sel_n64 - sel_pb` (what the unvetted Qwen scorer
adds over the vetted one), 95% paired bootstrap.

**Readings.** `gain(sel_pb)`: RETAINS (interval above 0) / NONE (contains 0) / HURTS (below 0). `DQ`: CONFIRMED /
UNRESOLVED / REFUTED as in feat-216.

**Prediction.** Under judge G, `gain(sel_pb)` RETAINS and `DQ` CONFIRMED: a fluency score keeps part of the gain.
No prediction under judge B.

**Gates.** G0: the cache covers 500 prompts x 64 ranks and every served text is non-empty where the pool has a
non-empty draw. G2: comparator levels equal the passes they are carried from.

**Excluded.** Any other scorer, normalisation, template or `n` chosen after a score is read.

## Scoring log
