# feat-216: does tempering the anchor recover selection's judged gain? (review 8, Q1 and roadmap 4)

Registered 2026-09-26 before any token of the new arms was drawn. Nothing above `## Scoring log` is edited afterwards.

**Question (review 8, Q1).** "What is the judged gain of the anchor alone at T in {0.5, 0.7} ... against best-of-64 at
T = 1.0?" Tempering is a per-token tilt of the untempered law, so if it alone buys selection's gain, the paper's
certified improvement is one a deployer could have had by choosing a different reference law.

**Arms (all against the committed opponent, `output/sweep_plain` `k=-1` rank 0, the 8B continuing text at 1.0).**
- `anchor_t07`: TinyComma alone (`k=0`) at temperature `0.7`, no repetition penalty. NEW.
- `anchor_t05`: the same at temperature `0.5`. NEW.
- `anchor_t07pen`: the same at `0.7` with repetition penalty `1.1`, the authors' book setting. NEW (the feat-195
  pool's trajectories were never pulled from host B, so it is drawn again).
- Comparators, the committed feat-210 pass: `sel_n64` (best-of-64 at 1.0, committed rewards), `sel_n1` (its rank-0
  draw, one untempered anchor draw), `anchor_k0` (`output/sweep_plain` `k=0`).

Generation: `h1.py --k-values 0 --trajectories-per-prompt 1 --seeds 52` with feat-210's caps (`--cap-neutral 200
--cap-factual 150 --cap-creative 150`, the rest `0`), `--batch-size 48`, the default TinyComma/8B pair, `T_max=200`,
`--temperature {0.7, 0.5}` and `--repetition-penalty {1.0, 1.1}` as above; `scripts/run_feat216.sh`, local A100s.
**Top-p truncation is not run**: the decoder (`a_patch/factory.py`) has no nucleus option and none will be added
under this deadline.

**Judging.** `analysis/matched_h2h.py`, tags `tempering_B` (Phi-3.5-mini, pre-specified) and `tempering_G`
(gemma-2-27b-it), both orders, local A100s, the machine type that judged the committed pass. The comparators'
verdicts are the committed pass's (`results/matched_h2h_verdicts_matched_plain_{B,G}.csv`), carried into the new
tag's cache; `matched_h2h.py` reuses a cached arm only when every text hash matches, so they are not re-judged.

**Quantities.** Every arm's control is `sel_n1`, so a difference `sel_n64 - anchor_T` is the paired difference of
order-averaged levels. Registered differences: `D07 = sel_n64 - anchor_t07`, `D05 = sel_n64 - anchor_t05`,
`D07p = sel_n64 - anchor_t07pen`, each with its 95% paired bootstrap interval (the script's), per judge.
Descriptive: each tempered arm's gain over `sel_n1`; `anchor_k0 - sel_n1` (two untempered draws, a null check);
empty completions per arm.

**Readings, per difference and judge.** `CONFIRMED` (interval above 0): selection adds to what tempering buys.
`UNRESOLVED`: tempering matches selection within the instrument. `REFUTED` (interval below 0): tempering beats it.

**Prediction.** `D07` and `D05` read `CONFIRMED` under judge G, whose two verdicts agree on 78 to 85% of items;
under judge B, pre-specified but order-consistent on a third of items, no prediction is made beyond reporting.

**Gates, before any reading.** G0: each new arm covers the 500 prompts, its records carry the registered temperature
and penalty, and every step is forced to the anchor (`k=0`). G2: the comparators' levels in the new pass equal the
committed pass's (`results/matched_h2h_matched_plain_{B,G}.csv`) exactly.

**Excluded.** Any other temperature, truncation, seed or judge chosen after a draw is read; pooling across
judges; comparing a level of this pass with a level of any other pass.

**Consequence for the manuscript, whatever the reading.** The closing's "Nor did we test whether tempering the
anchor recovers the gain" is replaced by the measured `D07` and `D05` under both judges, with Appendix~H carrying
the table; if a gate fails the sentence stays and the failure is reported here.

**Addendum, 2026-09-26 16:40, before any verdict of either judge.** Judge G at 27B needs two cards per job and about
ten minutes per arm on an A100, which the deadline does not allow for three arms. Judge G therefore runs on host B's
eight H100s, and to keep its comparison within one machine every arm is re-judged there, the comparators included
(tag `tempering_G`, per-arm caches merged before aggregation). G2 for judge G becomes descriptive: the comparators'
H100 levels are reported against the committed A100 levels, with no exactness required (caution (as): bf16 greedy
verdicts are not bitwise portable across GPU architectures). Judge B stays as registered (local A100s, comparators
carried from the committed pass). Nothing else changes.

## Scoring log

### Scored 2026-09-26 16:45 (`.venv/bin/python analysis/score_feat216.py` -> `results/tempering_scoring.csv`)

**Gates.** G0 PASS on all three arms (500 prompts each at the registered temperature and penalty, `k=0`, no step
active or risky). G2 PASS for judge B (comparator levels `0.555`, `0.4535`, `0.4385` equal the committed pass's).
G2 for judge G (descriptive, per the addendum): the H100 re-judging reproduces the committed A100 levels exactly,
`+0.0000` on all three comparators.

**Differences (best-of-64 at 1.0 minus the tempered anchor, paired levels, 500 prompts).**

| | judge B | judge G |
|---|---|---|
| `D07` (T=0.7) | `+0.0685 [+0.0455, +0.0915]` CONFIRMED | `+0.063 [+0.029, +0.096]` CONFIRMED |
| `D05` (T=0.5) | `+0.069 [+0.046, +0.0915]` CONFIRMED | `+0.060 [+0.0255, +0.0955]` CONFIRMED |
| `D07p` (T=0.7, penalty 1.1) | `+0.0365 [+0.0135, +0.0595]` CONFIRMED | `+0.051 [+0.0165, +0.085]` CONFIRMED |

**Prediction RIGHT**: `D07` and `D05` read CONFIRMED under judge G (and under judge B, where none was made).

**Descriptive.** Gains over one untempered draw (`sel_n1`): best-of-64 `+0.1015` (B) / `+0.176` (G); `T=0.7`
`+0.033` / `+0.113`; `T=0.5` `+0.0325` / `+0.116`; `0.7/1.1` `+0.065` / `+0.125`; the null (`anchor_k0`, a second
untempered draw) `-0.015 [-0.0365, +0.0065]` / `-0.0005 [-0.033, +0.0315]`. So tempering buys about a third of
selection's gain under judge B and about two thirds under judge G, and selection adds to it under both. **Empty
completions rise as the anchor is tempered**: `138` of `500` at `0.7`, `167` at `0.5`, `75` at `0.7/1.1`, against
`62` untempered and `41` for best-of-64; both judges score an empty answer near parity (Appendix B), so the tempered
gains are if anything flattering. Order consistency: judge B `0.31`--`0.38`, judge G `0.76`--`0.80` on the new arms.

**Manuscript.** Appendix `tab:tempering` (every cell from the two passes) and the closing sentence, which now reads
"Tempering buys only part of the gain: best-of-64 leads it at T=0.7 and 0.5 by +0.0685 and +0.069 (B), +0.063 and
+0.060 (G)", as registered. Guard: `tests/test_tempering.py`, mutation-checked on one closing number and one table
cell. **Not run**: the optional A100 cross-check of judge G on the `0.7` arm failed to start (the 27B judge is not in
this host's cache at the launcher's path); it was not a registered requirement once the addendum moved judge G to a
complete H100 pass.
