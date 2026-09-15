# Is the onset ratio a property of the corpus, or of the memoriser?

**Committed 2026-09-15 22:44, before any of these memorisers exists.** Nothing above
the scoring-log heading is edited after that line is written.

## The question feat-120 opened and could not answer

feat-120 changed the protected work for a second time and two of three committed bands failed: the
onset ordering inverted, with Pleias-1.2B rising from `0.878` and `0.895` on the first two corpora to
`1.3142` on BookMIA, and KL3M-520M's interval ceasing to exclude `1`. The paper retracted the claim
that leakage beginning after the certificate has gone vacuous is a property of the pair.

The file also recorded, in advance, the rival explanation it could not settle: Pleias' BookMIA
memoriser is much the weakest of the three, and a weak memoriser needs more budget before it can
leak. Its sampled `k=-1` recall falls `0.909 -> 0.517 -> 0.150` across the three corpora while its
ratio rises `0.878 -> 0.895 -> 1.314`. Over all nine (pair, corpus) cells the rank correlation is
only `rho = -0.317` at exact `p = 0.4101`, which is post hoc and not significant --- so the
explanation is neither supported nor excluded. **It cannot be settled by that design at all**,
because every memoriser is fine-tuned on the corpus it is then measured against, so corpus and
memoriser strength move together by construction.

One further observation, from the run log rather than the result, sharpens the question. The BookMIA
Pleias memoriser ran its full 40 epochs and finished at token loss `0.1098` --- its *worst* of the
last three (`0.0493` at 38, `0.0768` at 39, `0.1098` at 40). A LoRA at `lr 3e-4 rank 128` oscillates
near convergence, so its weakness is substantially an artefact of where a fixed epoch budget happened
to stop. That is a lever: it means strength can be varied on one pair and one corpus without
touching anything else.

## The design: one pair, one corpus, one knob

Pleias-1.2B on BookMIA --- the exact cell that inverted. Three new memorisers on the **identical**
`data/bench/bookmia100_onset600.jsonl`, with every flag identical to feat-120's except `--epochs`:

```
--target-modules all-linear --no-chat --lr 3e-4 --rank 128 --batch 2 --accum 4
--max-len 0 --stop-loss 0.02        --epochs 10 | 20 | 30
```

The fourth point is feat-120's own `--epochs 40` run, already measured, whose sweep and bootstrap
are on record. Four points, one pair, one corpus, one architecture, one training set.

**The knob does not need to be monotone and no band below assumes it is.** Strength is *measured*,
by each memoriser's own sampled `k=-1` arm on the swept passages, and every band correlates the
ratio against that measurement rather than against the epoch count.

The sweep grid is BookMIA's committed grid with feat-120's licensed extension already included, so
all four points share one grid and no crossing can sit at a ceiling:
`-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2 4.6 5.3 6.6`, `--modes single --limit 100`, on the
same 100 passages and the same seed.

## Committed bands

Primary metric: Spearman `rho` between each point's **sampled `k=-1` recall** and its onset ratio,
over the four points, with the exact permutation `p`. At `n = 4` the floor is `1/12 = 0.083`, so no
result here can be significant at `0.05` and none is claimed to be; the bands are about effect size.

| outcome | reading |
|---|---|
| `rho <= -0.8` **and** the four ratios span `>= 0.15` | **strength explains it.** The onset ratio is a function of memoriser strength within a fixed pair and corpus |
| the four ratios span `< 0.10` while the measured strengths span `>= 3x` | **strength does not explain it.** feat-120's retraction stands unqualified and the corpus is doing the work |
| anything else | **inconclusive**, reported as inconclusive. No second ladder, no fifth point |

**Entry gate, unchanged.** A point enters only if its sampled `k=-1` recall is at least `0.10`
(caution (a)). A point below it is excluded and reported as excluded --- that is a statement about
that memoriser, not about the question.

**A band on the arm's own validity, committed because it can void the result.** If the four measured
strengths span less than `3x` (max over min of sampled `k=-1`), the knob failed to produce a ladder
and the arm is **uninformative by construction**, reported as such whatever `rho` reads. An arm that
cannot separate its levels cannot answer its question.

## What each outcome licenses, written down before the numbers exist

This is the half that matters, because the obvious reading of a "strength explains it" result is
that feat-120 was wrong and the retracted sentence can come back. **It cannot, and the opposite is
closer to true.**

- If **strength explains it**, then the onset ratio is confounded with memoriser strength *in
  general* --- and the nine-pair CopyBench table, whose memorisers were never strength-matched
  (sampled `k=-1` runs from `0.2696` to `0.9091` across the arms on record), inherits that
  confound. The tokenizer split would then be a claim resting on nine differently-trained
  memorisers, which is **a larger problem for the onset section than feat-120's failure**, not a
  rescue of it. The retracted sentence stays retracted and Limitations gains the confound.
- If **strength does not explain it**, feat-120's reading is clean: the corpus moved the ratio, the
  retraction stands exactly as written, and the nine-pair table is not strength-confounded either.
- If **inconclusive**, nothing changes in the paper and this file is cited as the attempt.

**No outcome of this arm restores the retracted sentence.** It was retracted because the ordering
inverted on a third corpus, and no measurement of memoriser strength on one pair can un-invert it.

**What will not happen.** No second grid, no re-threshold, no re-seed, no fifth epoch count chosen
after seeing four, no swapping to another pair if Pleias is unhelpful.

## Scoring log
