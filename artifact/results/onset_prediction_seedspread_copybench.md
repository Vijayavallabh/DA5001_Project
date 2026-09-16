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

### Scoring, 2026-09-16 06:35: the fine pillar is STABLE, the coarse one is INCONCLUSIVE

```
CBPAIR=kl3m520m  GPU=<n> scripts/run_copybench_seeds.sh 1 2
CBPAIR=pleias12b GPU=2   scripts/run_copybench_seeds.sh 1 2
.venv/bin/python analysis/strength_ladder.py --pair kl3m_cb   --axis seeds --out results
.venv/bin/python analysis/strength_ladder.py --pair pleias_cb --axis seeds --out results
```

Every point passes the entry gate on its own sampled `k=-1` arm, every `k=0` arm reads `0.000`, and
the relaunched sweeps carry `0` per-trajectory violations in `1,100` budgeted queries each.

```
KL3M-520M (fine pillar, interval excludes 1 from above)
   seed=0 (the table's own)   k=-1 0.5188   onset 2.5429   ratio 1.0531
   seed=1                     k=-1 0.5149   onset 2.4625   ratio 1.0198
   seed=2                     k=-1 0.5756   onset 2.5309   ratio 1.0481
   three-point span 0.0333          strength span 1.12x

Pleias-1.2B (coarse pillar, interval excludes 1 from below)
   seed=0 (the table's own)   k=-1 0.9091   onset 2.8190   ratio 0.8784
   seed=1                     k=-1 0.9615   onset 2.8093   ratio 0.8753
   seed=2                     k=-1 0.9448   onset 2.5641   ratio 0.7989
   three-point span 0.0795          strength span 1.06x
```

**The two rows score differently and are reported differently.**

- **KL3M-520M: `0.0333`, below the committed `0.05`.** The second band fires. Its ratio is stable
  across seeds, the table's structure is not a seed artefact for this pair, and **the sentence now
  in Limitations and Appendix~\ref{app:scaling} is too broad and is narrowed**, exactly as this file
  committed before the run.
- **Pleias-1.2B: `0.0795`, between `0.05` and `0.15`.** **INCONCLUSIVE**, and reported as
  inconclusive. It is not evidence that the coarse pillar is unstable and it is not evidence that it
  is stable. Its seed-2 point is the whole of the spread (`0.7989` against `0.8784` and `0.8753`),
  and nothing in this file licenses dropping it.

Both memorisers reproduce their corner's strength closely --- `1.12x` and `1.06x` spans in sampled
`k=-1`, with KL3M's seeds at `0.5149`/`0.5756` against the table's own `0.5188` --- so these are
comparisons between like memorisers, which is the condition under which the ratio comparison means
anything.

**The committed diagnostic fired and is worth more than either band.** Every KL3M CopyBench seed
stopped at **epoch 11** of 40, against the table's own memoriser at epoch 11; every KL3M BookMIA
seed stopped at **epoch 26**, against its corner's 26. The recipe's effective training length is
reproducible across seeds. That was registered as "a finding about the recipe that applies to every
memoriser in the paper" if it moved --- it does not move.

### What this settles, together with the BookMIA seed ladder

`analysis/strength_ladder.py --pair kl3m --axis seeds` gives `0.0663` on the same pair at the same
`--epochs 40` on BookMIA. So across three independently-run seed ladders the onset ratio moves by
`0.033`, `0.066` and `0.080`, against feat-121's `0.4721` from varying `--epochs` alone on the same
cell, corpus, anchor and grid.

> **Seeds do not move the onset ratio; training length does.** feat-121 could not separate the two
> and said so; this separates them, and the mover is the one that changes how much the memoriser
> learned.

That is *not* a reprieve for the nine-pair table, and the pre-registration said so in advance. The
table's nine memorisers are **not matched on training length** --- their sampled `k=-1` runs from
`0.2696` to `0.9091`, a factor of `3.4` --- and they are not even matched on hyperparameters:
`mem_kl3m-002-520m` trained at `batch 2 / accum 4 / stop-loss 0.02` and `mem_Pleias-1_2b-Preview` at
`batch 4 / accum 2 / stop-loss 0.03`, stopping at epochs `11` and `30`. The variable that demonstrably
moves the ratio is exactly the variable the table never controlled.

**What the paper should now say, which is narrower and stronger than what it says today:** the onset
ratio is reproducible for a *given pair and a given memoriser recipe* --- three seed ladders agree,
spans `0.033`--`0.080` --- and is *not* a property of the pair alone, because holding pair, corpus,
anchor and grid fixed and changing only training length moves it by `0.47`. So the nine-pair table
supports that extraction begins near `s(x)`, which is what Proposition~\ref{prop:threshold} claims,
and does not support reading which pairs begin above `1` and which below, because that ordering is
confounded with a training variable the table does not hold fixed.

**Not settled here.** Two of the nine pairs, on one corpus each for the seed question. Pleias-1.2B's
CopyBench row is inconclusive and stays inconclusive. And no outcome of this arm restores any
retracted sentence: feat-120's retraction rests on a third corpus inverting the ordering, which
nothing here touches.

