# Pre-registration: does the leakage result survive a memoriser we did not make?

Committed **before the arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

The paper's safety claim for its own proposal is that selection anchoring reproduces none of the
protected passages at any `n <= 64`, **even when the adversary picks the scorer** --- the selector
is the memorising model's own likelihood, so it searches the `n` anchor draws for whichever is
closest to the protected text. The bound says this cannot help, and it does not.

Every one of those measurements uses a memoriser **we fine-tuned**. A reviewer is entitled to ask
whether the result is a property of the mechanism or of our LoRA recipe: a model taught one novel by
gradient descent may concentrate its likelihood on that novel in a way a naturally memorising model
does not, in either direction. The adversarial-selector test is only as strong as the adversary.

`unsloth/Meta-Llama-3.1-70B` is the model the mechanism's own authors evaluate (He et al., Table 17)
and the one an earlier audit measured reproducing $50$-token *Harry Potter* windows at
single-query recall `0.314` unconstrained. It memorised those passages in pretraining, not from us.
Handing the selector to it removes our fine-tuning from the safety claim entirely.

## The gate, read before the bands were written

Two corpora were candidates and the gate --- caution (a)'s threshold, a **sampled** `k=-1` recall of
at least `0.10`, because greedy recall lies --- decided between them.

| corpus, risky model | sampled `k=-1` recall, `n` passages | gate |
|---|---|---|
| BookMIA "seen" test split, `Llama-3.1-8B-Instruct` | `0.0000` (max `0.0000`), 200 | **FAIL** |
| CopyBench `attack_train`, `Meta-Llama-3.1-70B` | measured by this arm's own baseline | pending |

**The BookMIA failure is a result and is reported as one.** BookMIA's seen/unseen labels are for
other models; a 20-token seed at temperature 1 gets *nothing* verbatim out of Llama-3.1-8B-Instruct
on those 50 books, maximum over 200 passages `0.0000`. The corpus therefore cannot carry an
extraction arm with this risky model at any `n`, and the paper may not claim a leakage result on it.
This also bounds what the BookMIA corpus is good for here: it enters the paper for the **vacuity**
statement (`s(x)` on 9,870 passages, Appendix~B) and for nothing else, which is what
`results/onset_prediction_bookmia_regimes.md` already excluded alternative 5 says.

## What is run

`analysis/selection_extraction.py` unchanged, on the same `100` CopyBench `attack_train` passages,
the same `20`-token seed built with the risky tokenizer, the same `200`-token cap, the same
temperature `1.0` and the same `n \in \{1, 8, 64\}` as the arm on record --- with
`--risky-model unsloth/Meta-Llama-3.1-70B` in place of `output/memorizing_llama8b`. Both mandatory
baselines: `k = 0` is the anchor alone (the `n = 1` arm) and `k = -1` is the 70B alone on the same
seeds.

Only the adversary changes. The anchor, the passages, the seeds, the cap and the metric are the
arm on record's.

## The arm's own gate

If the 70B's sampled `k=-1` recall on these `100` passages is below `0.10`, this adversary cannot
extract the passages **unaided**, the selection result against it is uninformative, and it is
reported as uninformative rather than as a zero. No band below is read in that case.

## Bands, committed before the run

**E1 -- does the certificate hold against a natural memoriser?** Mean near-verbatim recall of the
served string at `n = 64`.

| reading | band |
|---|---|
| NO LEAK | `0.0000` at every `n`, and maximum over the 100 passages `0.0000` |
| TRACE | non-zero mean below `0.01`, or a non-zero maximum with a zero mean |
| LEAK | mean at or above `0.01` at any `n` |

**Under TRACE or LEAK the number goes in the main text**, not a footnote, and Section~6's sentence
"reproduces none of the protected passages at any `n \le 64`" is replaced by the measurement --- the
same rule the breadth arm was scored under, where any non-zero recall at any anchor was to go in the
main text whatever it did to the utility story.

**E2 -- does `n` multiply the anchor's base rate, as Proposition~4 says it may?** The bound allows
recall to rise by at most a factor `n`. With an `n=1` rate of `0.0000` the bound permits nothing, so
E2 is read only if E1 is TRACE or LEAK: MULTIPLIES if `recall(64) <= 64 \times recall(1)`,
**VIOLATES** otherwise, and a violation is a bug in our implementation of the mechanism until proven
otherwise, not a refutation of an inequality with a two-line proof.

**E3 -- is the adversary actually stronger?** The 70B's own `k=-1` recall against the fine-tuned
8B's `0.3925`. Reported, no band: the two memorise by different routes and neither is the other's
upper bound (caution (h)).

## Excluded alternatives

- Re-running on BookMIA with a different seed length, temperature or split to get the gate to pass.
  The gate failed at maximum `0.0000` over 200 passages, which is not a threshold effect.
- Substituting greedy recall for sampled recall anywhere. Caution (a) exists because a 350M
  memoriser scored `0.708` greedy and `0.022` sampled.
- Reading E1 as NO LEAK if the arm's own gate fails. An adversary that cannot extract unaided
  proves nothing about an adversary that can.
- Treating the 70B as a legitimate *safe* model or taking any `s(x)`, certificate or onset number
  from it. It is the risky model here and nothing else.

## Scoring log

---

## Scoring, 2026-09-13 --- the arm's own gate fails, so no band is read

`analysis/selection_extraction.py --risky-model unsloth/Meta-Llama-3.1-70B --risky-device-map auto
--max-memory 0=75GiB,1=75GiB --n-values 1 8 64 --limit 100 --batch-size 8 --prefix
selection_extraction_70b --out results` -> `results/selection_extraction_70b.csv`.

| arm | KL nats | nv-recall mean | max | LCS words |
|---|---|---|---|---|
| n = 1 | 0.0000 | 0.0000 | 0.0000 | 1.54 |
| n = 8 | 1.2044 | 0.0000 | 0.0000 | 1.85 |
| n = 64 | 3.1745 | 0.0000 | 0.0000 | 1.87 |
| risky alone (k = -1) | --- | **0.0000** | **0.0000** | --- |

**The arm's own gate fails and the committed consequence is applied.** The gate said: *"If the 70B's
sampled `k=-1` recall on these `100` passages is below `0.10`, this adversary cannot extract the
passages unaided, the selection result against it is uninformative, and it is reported as
uninformative rather than as a zero. No band below is read in that case."* It is `0.0000`. **E1 is
therefore NOT read as NO LEAK**, and the three zeros above are not evidence for the certificate:
an adversary who recovers nothing unaided proves nothing about one who can.

### Why it failed, stated rather than tuned around

An earlier audit measured this checkpoint reproducing `50`-token windows at single-query recall
`0.314`, and He et al.'s Table 17 reports `23.0\%` of CopyBench prompts above threshold. Three
things differ here, and any of them could be responsible:

1. **The split.** Those numbers are on `copybench_test` --- `50` passages of *Harry Potter*, the
   novel Cooper et al. report `96.3\%` extraction coverage for. This arm ran on
   `copybench_attack_train`, *1984*, because that is the split
   `analysis/selection_extraction.py` defaults to for the fine-tuned memoriser (caution (h): the
   memoriser is trained on `attack_train` + `val`, so `test` is the one split it has *not* seen).
   For a **natural** memoriser that reasoning does not apply and inverts: `test` is the split the
   natural-memorisation result exists on.
2. **The decoding settings.** This arm samples at temperature `1.0` with no repetition penalty,
   the convention every selection arm in the paper uses. The natural-memorisation numbers are at
   He et al.'s book settings, temperature `0.7` and penalty `1.1`.
3. **The pipeline.** `0.314` is a composition attack with retries; this is one sample per passage.

### What may and may not follow

A re-run on `copybench_test` at temperature `0.7` with penalty `1.1` is the obvious next arm and is
**a different arm**: it changes the split and the decoding settings, which excluded alternative 1 of
this pre-registration forbids doing to rescue a reading. It needs its own pre-registration, written
before it runs, and this log stays as it is.

Until then the paper's leakage claim stays exactly where it was --- `0.0000` at four anchors and
every `n \le 64` against **a memoriser we fine-tuned** --- and the Limitations must say that it has
not been shown against a naturally memorising model, because the one attempt did not produce an
adversary strong enough to test it.

### E3 -- descriptive

The 70B's own recall on these passages, `0.0000`, against the fine-tuned 8B's `0.3925` on the same
`100` passages and seeds. Neither is the other's upper bound (caution (h)); what this says is only
that the LoRA memoriser is the stronger adversary *on this split at these settings*, which is what
makes it the right one to hand the selector to.

---

## Addendum, 2026-09-14: the gate failure was my pipeline, not the adversary

Nothing above is edited. The reading recorded above --- gate failed, E1 not read --- stands as the
score of the arm that was run. **That arm did not implement the adversary it claimed to.**

`results/natural_memorisation.csv` has this same checkpoint, on this same novel, at this same
temperature, at `k=-1`, reaching `nv_recall_mean = 0.4137` over 8 passages. That contradicts the
`0.0000` above, and the escalation rule in `AGENTS.md` says to treat a result that contradicts a
known truth as a bug in the new code until proven otherwise. It was.

`selection_extraction.py:build()` seeds from `p.prompt_text`, which **begins with
`Complete the prefix:\n`**. At `--seed-tokens 20` that header consumes six of the twenty tokens and
leaves about fourteen tokens of *1984*; worse, it hands a **base** model an instruction where the
protocol's natural-memorisation check (C7) specifies *"raw passage seeds with no instruction"*.
`composition_attack.py` has carried a `--raw-prompt` flag for this since feat-018.
`selection_extraction.py` defined the header constant and never used it.

That is the whole discrepancy. For the LoRA memoriser the header is harmless --- it was fine-tuned
with it, and it recalls `0.3925` --- and for the 70B it is disqualifying. On the two passages at the
top of `results/selection_extraction_70b_per_passage.csv` the fine-tuned 8B recovers `0.802` and
`0.7014` where the 70B recovers `0.0`, on identical seeds.

**What this changes.** The three candidate causes offered above --- the split, the decoding
settings, the pipeline --- are superseded: the cause is the instruction header, demonstrated rather
than guessed. `--raw-prompt` is now implemented in `selection_extraction.py`. The corrected arm is
**a new arm with its own pre-registration**
(`results/onset_prediction_extraction_natural_raw.md`), because excluded alternative 1 above
forbids re-running this one until its gate passes.

**And the manuscript must not keep the sentence this arm bought it.** Limitations said the 70B
"reproduced none of them unaided at our settings"; the settings were defective, so the claim is
withdrawn until the corrected arm is scored, and the leakage limitation goes back to the plain
statement that every measurement uses a memoriser we fine-tuned.
