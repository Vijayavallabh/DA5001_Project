# The seed test, run on the table the claim is actually about

**Committed 2026-09-16 03:36, before either memoriser exists** and before any seed result on any
pair exists --- the two BookMIA seed arms are still in their fine-tunes, with no sweep begun.
Nothing above the scoring-log heading is edited afterwards.

## Why this pair and this corpus

Every seed and epoch measurement so far --- feat-121 and both running arms --- is on **BookMIA**.
The claim they are used to qualify is about the **nine-pair CopyBench table** of Section 4: that its
ratios are not readable as per-pair properties. That is an argument by analogy, and it can be
replaced by a direct measurement.

`KL3M-520M + mem. KL3M-520M` is the right pair, and it is the adversarial choice rather than the
convenient one. It is **in** the nine-pair table; it is one of the two pairs whose interval excludes
$1$ (`1.0532` `[1.0163, 1.2436]`), which is the evidential basis for "leakage begins after the
certificate has gone vacuous"; it is the fine-tokenizer pair the whole tokenizer split is built on;
and its no-crossing fraction there is `0.0%`, so there is no grid-ceiling escape hatch. **If its
ratio is stable across seeds, the paper's new sentence is too broad and must be narrowed.**

## The protocol, recovered and then PROVEN rather than reconstructed

The original run's command is in no surviving log, which is exactly the situation caution (v) warns
about. Two artefacts make it recoverable anyway, and the second was checked rather than assumed:

1. `output/phase5/mem_kl3m-002-520m/recipe.json` records the fine-tune in full: base and tokenizer
   `alea-institute/kl3m-002-520m`, `splits ['attack_train', 'val']`, `epochs 40` (ran `11`, stopping
   on `--stop-loss 0.02` at `final_loss 0.0198`), `rank 128`, `lr 3e-4`, `batch 2`, `accum 4`,
   `no_chat True`, `target_modules all-linear`, `max_len 679` (the resolved value of `--max-len 0`),
   `seed 0`, `n_texts 608`.
2. The sweep's corpus is recorded per passage in `output/phase5/fine_kl3m520m/composition.csv`:
   $100$ prompt ids over three novels --- $50$ *A Game of Thrones*, $42$ *Casino Royale*, $8$
   *1984* --- which is the composition AGENTS.md caution (v) documents for
   `--split attack_train --limit 100`. **That reconstruction was verified, not inferred**: rebuilding
   the selection through `dap.shared.load_prompt_corpus("data", "factscore_prompt")` with
   `p.split == "attack_train"`, a non-empty reference and `--limit 100` reproduces the swept ids
   **exactly and in the same order**. Anything short of an exact match would have made this arm
   unbuildable and it would not have been run.

The grid is the pair's own committed grid, taken from its summary rather than chosen:
`-1 0 1.6 1.8 2.0 2.1 2.2 2.3 2.4 2.6 2.8 3.0 3.4`, `--modes single --limit 100`, and
$s(x) = 2.41475$ from `results/onset_ci.csv`.

Two new memorisers at `--seed 1` and `--seed 2`, differing from the table's own memoriser in the
seed and nothing else. Its `seed 0` run is the third point, already measured and bootstrapped.

## Committed bands

The quantity is the **three-point seed span** over `{0, 1, 2}` --- the same $n$ as the committed
primary of both BookMIA arms, so all three pairs are compared like with like and no span is read
against a span of different size.

| outcome | reading |
|---|---|
| span **`>= 0.15`** | the nine-pair table's own ratios are not reproducible across seeds. Its between-pair spread of `0.2874` is then partly a seed artefact, measured on the table itself rather than by analogy, and the paper says so in Section 4 and not only in Limitations |
| span **`< 0.05`** | this pair's ratio is stable, the table's structure is not a seed artefact for it, and **the sentence now in Limitations and Appendix~\ref{app:scaling} is too broad and gets narrowed to the pairs and corpus it was measured on** |
| in between | **inconclusive**, reported as inconclusive, with the span quoted beside both BookMIA arms' |

**Entry gate, unchanged.** A point enters only if its sampled `k=-1` recall is at least `0.10`
(caution (a)). The table's own `seed 0` point reads `0.5188`. A point below the gate is excluded and
reported as excluded.

**Reported either way, as a diagnostic and not a band:** how many epochs each seed runs before
`--stop-loss` fires. The table's memoriser stopped at `11` of `40`. If that number moves with the
seed, then "40 epochs" does not fix the amount of training, which is a finding about the recipe that
applies to every memoriser in the paper.

## What this can and cannot do

It measures one pair of the nine on the corpus the table is built from. It cannot separate seed from
memoriser strength any better than the BookMIA arms can, and it is not a replacement for them; it
removes the *analogy* step, not the confound. Two seeds is the most the open compute window allows,
and no third will be added: the window closes at about `09:13` and this file is written to be scored
on three points or not at all.

**No outcome of this arm restores any retracted sentence.** feat-120's retraction rests on a third
corpus inverting the ordering, which nothing here touches.

**What will not happen.** No re-grid, no re-threshold, no third seed, no swapping to an easier pair,
and no dropping a seed that lands awkwardly.

## Scoring log

### Extension, 2026-09-16 03:53: a second row of the same table, on GPU 2

At `03:51` the user directed that GPU 2 be used as well. This file's design is extended to a
**second pair of the nine-pair table**, `Pleias-1.2B + mem. Pleias-1.2B`, under identical rules and
with the same three-point primary. Recorded here rather than by editing above the scoring log.

**Why this pair.** KL3M-520M is the *fine*-tokenizer pillar of the split. Pleias-1.2B is the
*coarse* pillar: ratio `0.8784` `[0.7891, 0.9600]`, interval excluding `1` from below, no-crossing
`0.0%`. Between them they are the two sides the tokenizer split is stated as. If both are stable
across seeds, the table's structure is not a seed artefact and the sentence now in Limitations and
`appendix_robustness.tex` is too broad; if both move, it is, measured on the table itself.

**No result of any kind exists** at the time of writing: not one of the seven running jobs has
produced a sweep, so this extension cannot be a response to an answer.

**Its protocol is its own, and that is a finding in itself.** The nine memorisers of the table are
**not hyperparameter-matched**: `mem_kl3m-002-520m` trained at `batch 2 / accum 4 / stop-loss 0.02`
and `mem_Pleias-1_2b-Preview` at `batch 4 / accum 2 / stop-loss 0.03`, with `max_len` `679` against
`448` and `epochs_run` `11` against `30`. Each ladder therefore inherits *its own* corner's recipe,
read from that corner's `recipe.json`; a ladder trained on the other pair's settings would not be a
ladder on that corner at all. That the table's rows differ in batch, accumulation and stopping rule
is worth stating in the paper independently of how these arms score.

Its corpus was proven the same way as KL3M's: rebuilding `--split attack_train --limit 100` through
`load_prompt_corpus` reproduces `fine_pleias12b/composition.csv`'s prompt ids **exactly and in
order**. Grid `-1 0 2 2.4 2.6 2.7 2.8 2.9 3 3.2 3.6`, its own, from `results/onset_ci.csv`;
$s(x) = 3.2094$.

**Bands, entry gate, and everything else above are unchanged and apply to this pair too.** Two
seeds, `1` and `2`, plus the table's own `seed 0`: a three-point primary, the same `n` as every
other arm. The same prohibitions hold — no third seed, no re-grid, no re-threshold, no swapping
pairs, no dropping a seed that lands awkwardly.
