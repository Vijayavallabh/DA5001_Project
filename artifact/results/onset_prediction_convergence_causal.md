# Is non-convergence the *cause* of the irreproducible onset ratio, or a companion of it?

**Committed 2026-09-16 10:40, before any of these memorisers exists.** Nothing above the
`## Scoring log` heading is edited after that line is written.

## The claim this tests is one we wrote into the paper this morning

Four seed ladders are on record. Three reproduce (ratio spans `0.0333`, `0.0663`, `0.0795`); one
does not (`0.2597` over three seeds, `0.3469` over four). The appendices now say the ratio is
reproducible *wherever the fine-tune converged and the memoriser is strong*, and label that reading
**post hoc** — because the one irreproducible cell, Pleias-1.2B on BookMIA, is also the only one
whose stop-loss never fires and the only one whose memoriser barely clears the entry gate. Three
things move together and nothing separates them.

They are separable, because convergence here is a **manipulable** variable rather than a property of
the pair. At `lr 3e-4` this cell's final loss is not monotone in training length — `0.0623` at 10
epochs, `0.0865` at 20, `0.0258` at 30, `0.1098` at 40 — and across four seeds at 40 epochs it runs
`0.0619` to `0.1143`, never reaching the `0.02` stop-loss. That is an optimiser oscillating, not a
model still learning, which AGENTS.md already records for this recipe. Lower the learning rate and
nothing about the pair, the corpus, the anchor or the grid changes; only the fine-tune's stability
does.

## Design

**Stage 1, the probe.** Three fine-tunes at `--lr 1.5e-4`, `1e-4`, `5e-5`, seed 0, everything else
identical to the cell's own recipe (`--epochs 40 --rank 128 --batch 2 --accum 4 --stop-loss 0.02
--target-modules all-linear --no-chat --max-len 0`, corpus `bookmia100_onset600.jsonl`).

> **The learning rate carried into stage 2 is the LARGEST probed rate whose stop-loss fires.**
> Committed now so it is not chosen after seeing any ratio. Largest, because the intervention should
> be the smallest departure from the published recipe that achieves convergence.

**Stage 2, the ladder.** Seeds `1` and `2` at that rate, joining the probe's seed 0 for three points,
swept on the cell's own committed grid
`-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2 4.6 5.3 6.6`, `--modes single --limit 100`, on the
same 100 passages.

## Committed bands

The quantity is the **three-seed ratio span**, read against the same cell's three-seed span at the
published `lr 3e-4`, which is `0.2597`. Three seeds against three seeds: like for like.

| outcome | reading |
|---|---|
| span **`< 0.10`** | **convergence is the cause.** The ratio is reproducible once the fine-tune converges, on the very cell where it was not, with the pair, corpus, anchor, grid and passages fixed. The paper's explanation stops being post hoc and becomes an intervention |
| span **`>= 0.20`** | **convergence is not the cause.** A converged memoriser on this cell is still irreproducible, so the explanation now in `appendix_robustness.tex` and `appendix_limitations.tex` is withdrawn and the irreproducibility is attributed to the pair, not to the optimiser |
| anything else | **inconclusive**, reported as inconclusive. No fourth seed, no second learning rate |

**Entry gate, unchanged.** A point enters only if its sampled `k=-1` recall is at least `0.10`
(caution (a)). A point below it is excluded and reported as excluded.

**A diagnostic, not a band, reported either way:** the strength span, and whether every seed's
stop-loss fires and at which epoch. A lower learning rate may produce a *weaker* memoriser, and
strength is the variable the ratio tracks. If the new ladder's sampled `k=-1` lands far outside the
`0.150`–`0.239` the published-rate ladder occupied, then the intervention moved convergence **and**
strength, the two are again not separated, and that is disclosed as a confound rather than reported
as a clean result.

## What makes this arm INVALID rather than a failure

A defect in our own construction must not be allowed to answer the question (caution (w)):

- **No probed rate converges.** Then we failed to build the intervention. INVALID; not evidence that
  convergence is irrelevant.
- **The converged memoriser falls below the entry gate.** Then the intervention destroyed the
  memorisation it was supposed to stabilise, and the arm compares a memoriser against a non-memoriser.
  INVALID.

## What will not happen

No second learning-rate sweep after seeing a ratio, no fourth seed, no re-grid, no re-threshold, no
dropping a seed that lands awkwardly, and no reporting of stage 2 if stage 1 leaves the arm invalid.

## Scoring log

## Scoring, 2026-09-16 --- stage 1 INVALID by this arm's own rule. No rate converged in 40 epochs.

```
bash scripts/run_convergence_probe.sh <lr> <gpu>     # 10:38-11:28, GPUs 0/1/4
```

| probe | epochs | final loss | stop-loss | converged? |
|---|---|---|---|---|
| `lr 1.5e-4` | 40/40 | `0.0229` | 0.02 | no |
| `lr 1e-4` | 40/40 | `0.0260` | 0.02 | no |
| `lr 5e-5` | 40/40 | `0.0305` | 0.02 | no |

**None crossed the stop-loss, so by the condition written down before the runs this stage is
INVALID: we failed to build the intervention.** It is not evidence that convergence is irrelevant,
and stage 2 is not run on it. That was the whole point of writing the condition down.

### What failed, and what did not

The rates were not wrong. They removed exactly the thing they were chosen to remove. At the
published `lr 3e-4` this cell's loss is wildly non-monotone in training length --- `0.0623` at 10
epochs, `0.0865` at 20, `0.0258` at 30, `0.1098` at 40 --- and across four seeds at 40 epochs it
runs `0.0619` to `0.1143`. At `lr 1.5e-4` the last four epochs read `0.0269`, `0.0254`, `0.0236`,
`0.0229`: **monotone, smooth, and still descending when the epoch cap stopped it.** The final
losses are also monotone in the rate (`0.0229 < 0.0260 < 0.0305` for `1.5e-4 > 1e-4 > 5e-5`), which
is what stable descent looks like and what the published rate does not do.

So the optimiser was stabilised and the memoriser was left short of the threshold. The diagnosis
points at the **epoch cap**, which this arm fixed at the recipe's own `40` and did not vary.

### What happens next, and why it is not a second bite

A new arm, `results/onset_prediction_convergence_causal_60.md`, re-probes the same three rates under
the same selection rule with the cap raised to `60`.

**No ratio has been measured.** Nothing has been swept, no onset exists, and this stage produced
four loss numbers and nothing else. Re-attempting a construction that demonstrably failed to
construct is not the same act as re-running an experiment whose answer one dislikes, and the
difference is exactly that: an outcome has not been seen. Had a single seed been swept, the
honest course would be to stop.

The original bands, the selection rule, the entry gate and both invalidity conditions carry over
unchanged into the new arm. The only edit is the epoch cap, and it is made for a stated reason that
is visible in the loss curve above rather than in any result.
