# Pre-registration: a second judge-free head-to-head, on a task the weak anchor can actually do

Committed **before the corpus is built or any generation runs**. Nothing above the
`## Scoring log` line is edited afterwards.

## Why, and what the two previous attempts taught

The judge-free comparison against a metered decoder exists on **one** task, TriviaQA. It needs an
anchor that shares the risky model's tokenizer --- only TinyComma-1.8B does --- and that anchor
cannot do the other two we tried: `0.04` on GSM8K, and **below chance** on MMLU under a repaired
parser (`onset_prediction_mmlu_rescore.md`). Neither failure is really about capability. Both tasks
ask a small base model trained on public-domain text to follow an *instruction format*.

**LAMBADA asks it to finish a sentence**, which is what a base language model does natively: the
prompt is a passage minus its last word and the answer is that word. There is no few-shot prefix,
because the task is the format.

Two defects from the MMLU arms are repaired here **in advance**, not after:

1. **The parser is pinned to output shapes before the run.** `tests/test_lambada_extraction.py`
   checks it against a bare continuation, a run-on, quoting and punctuation, and an empty
   completion --- against the *format*, never against which answer is right.
2. **Both gates below have a direction.** MMLU's H1 read ``the interval excludes `0.25` '' and the
   anchor excluded it *downward*; read literally that gate passed on a model performing at a third
   of chance.

## What is run

`analysis/build_bench_corpora.py::build_lambada` writes `data/bench/lambada_factual.jsonl` (the
first `500` items of `EleutherAI/lambada_openai`, English) and the symlink data-dir
`data/bench/lambada`, from the same loader the selection arm uses. Then, exactly as the TriviaQA
head-to-head does:

```
h1.py --k-values -1 0 0.5 1 3 20 --trajectories-per-prompt 1  --data-dir data/bench/lambada ...
h1.py --k-values 0            --trajectories-per-prompt 64 --data-dir data/bench/lambada ...
analysis/verifiable_metered.py --task lambada --corpus data/bench/lambada_factual.jsonl
```

anchor TinyComma-1.8B, risky `Llama-3.1-8B-Instruct`, `--max-new-tokens 8` (one word plus run-on),
`--cap-factual 500`. `k=-1` and `k=0` are in the sweep, as every arm here requires.

## Bands, committed before the run

**H1a --- the instrument check, read first.** The unconstrained risky model (`k=-1`) must reach at
least **`0.40`** exact match. `Llama-3.1-8B` is reported near `0.70` on LAMBADA; a reading far
below that means the parser or the prompt is wrong, whatever the anchor does. **If H1a fails,
nothing else is read** and the arm is invalid rather than negative.

**H1b --- the entry gate, with a direction.**

| reading | band |
|---|---|
| ANCHOR CAN DO THE TASK | the `n=1` / `k=0` accuracy 95% interval lies **strictly above** `0.05` |
| TOO WEAK | the interval contains or lies below `0.05` |

`0.05` rather than a chance floor because LAMBADA is open-vocabulary: guessing scores ~`0`, so any
real signal is well above it. **We predict ANCHOR CAN DO THE TASK**, and if we are wrong the arm is
reported as a third instance of ``where the anchor cannot do the task, no `n` helps'' and the
judge-free head-to-head **stays a one-task result**, said plainly.

**H2 --- does selection lift?** `SELECTION LIFTS` if majority vote's gain at `n=64` over `n=1` is
positive with its interval excluding zero; `FLAT` if it contains zero.

**H3 --- the frontier, which is the claim.** For each accuracy selection reaches over the shared
control, `selection nats / cheapest metered nats reaching it`. FRONTIER HOLDS if every ratio is
`< 1`; MIXED if some are `>= 1`; FRONTIER FAILS if the median is `>= 1`. Selection's figure is the
closed-form bound and the meter's its **measured** mean spend, so every ratio runs in the
conservative direction and the paper says so.

On TriviaQA the ratios are `0.032, 0.061, 0.093, 0.056, 0.071`. **We predict FRONTIER HOLDS.**

## H4 --- the manuscript consequence, fixed now

- **H1a fails.** Invalid, not negative. No number enters the paper and we do not get a second
  parser repair on this corpus.
- **H1b TOO WEAK.** One sentence beside the two chanceless anchors; the judge-free head-to-head
  remains one task and Limitations keeps saying so.
- **H1b passes, FRONTIER HOLDS.** Section 4.3's judge-free head-to-head names **two** tasks and the
  Limitations sentence about a single one is struck.
- **H1b passes, MIXED or FRONTIER FAILS.** Reported in the **main text** beside TriviaQA. A second
  task that disagrees with the first is the most informative outcome available and is not buried.

## What may not be claimed

No judged number, no certificate, no leakage, no `s(x)`: LAMBADA passages are not protected text
and every cap here is `k\,T_max` at `T_max = 8`. No accuracy level is quoted beside TriviaQA's ---
different tasks, different floors --- only the ratios, and only as ratios.

## Excluded alternatives

- Repairing the parser after seeing a result, or adding a few-shot prefix to raise the anchor.
- Changing `T_max`, the item count or the `k` grid after any reading.
- Reporting H2 or H3 if H1a fails or H1b reads TOO WEAK.

## Scoring log

## Scoring, 2026-09-20 --- H1a FAILS, the arm is INVALID, and there is no second attempt

Run completed (`results/verifiable_metered_lambada.csv`, generations under
`output/phase5/lambada_{metered,sel64}`). **No band below H1a is read.**

### H1a --- the instrument check: FAILS

The unconstrained risky model (`k=-1`, `Llama-3.1-8B-Instruct`) reads **`0.102`** `[0.076, 0.130]`
against the registered floor of **`0.40`**, and a published LAMBADA near `0.70`. Under this
document's own H4 that makes the arm **invalid rather than negative**, and no number from it enters
the paper.

### The cause, and it is the one this arm was built to avoid

The models **echo the prompt's tail before continuing**:

```
served prompt tail  ... it's supposed to make the wood burn longer." "But why do I have to
generation          ' "But why do I have to be the one to sing?" "Well'
gold                sing
```

The target word is present in the continuation; it is simply not the **first** word, and
`extract_lambada` takes the first word. That is the **same defect that invalidated both MMLU arms**
(`onset_prediction_mmlu_headtohead.md`: a base model echoing the option list, `301` of `500` scored
wrong).

### What I got wrong, stated plainly

This document claims the parser was ``pinned to output shapes before the run'', and
`tests/test_lambada_extraction.py` does pin it --- to a bare continuation, a run-on, quoting,
punctuation and an empty string. **Every one of those shapes I imagined.** The shape that actually
occurs, and the one that had already broken two arms on this project, is the echo, and I did not
test for it because I had not looked at a single generation from these models on this prompt type.

Testing a parser against hypothesised formats is not the MMLU lesson. The lesson is: **run a few
prompts first and read the output, then write the parser against what the models actually emit.**
Inspecting generations for their SHAPE costs nothing and reveals no result --- it is not peeking,
because the accuracy is not computed. I applied half the lesson and the arm failed for the half I
skipped.

### Consequence, applied

- The arm is invalid. Per H4, **there is no second parser repair on this corpus**, and none is
  attempted. `results/verifiable_metered_lambada.csv` is kept only as the record of the failure.
- The judge-free head-to-head against a metered decoder therefore **remains a one-task result**,
  TriviaQA, and Limitations continues to say so.
- Three tasks have now been tried at the one meterable anchor and all three failed for reasons that
  are ours or the anchor's, not the mechanism's: GSM8K (anchor scores `0.04`), MMLU (anchor below
  chance), LAMBADA (parser cannot read an echo). That is worth recording as a bound on what this
  head-to-head can be extended to without a different anchor.
