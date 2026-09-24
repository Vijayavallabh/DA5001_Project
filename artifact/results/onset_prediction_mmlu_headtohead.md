# Pre-registration: a second judge-free head-to-head, on MMLU

Committed **before the corpus is built or any generation runs**. Nothing above the
`## Scoring log` line is edited afterwards.

## Why this arm exists

The judged head-to-head split under a judge panel: on byte-identical text the order-averaged paired
difference reads `+0.0645` (judge B), `+0.0620` (judge D) and `+0.0090` (judge E), so two of three
judges resolve it and one does not (`results/onset_prediction_frontier_judge.md`). The paper has
been demoted accordingly, to *selection buys at least the meter's utility for one fifty-fourth of
its divergence*.

That demotion puts weight on the **judge-free** axis, and there the head-to-head against a metered
decoder exists on exactly **one task**: TriviaQA at TinyComma-1.8B (`verifiable_metered_tqa.csv`).
One task is not an axis. This adds a second.

## Why MMLU, and why not a third open-ended task

The head-to-head needs an anchor that shares the risky model's tokenizer --- only TinyComma-1.8B
does --- and that anchor must be able to do the task. It scores `0.04` on GSM8K and `0.112` on
TriviaQA. MMLU is **four-way multiple choice**, so the floor is `0.25` rather than `0.00`, and a
weak anchor's signal above chance is measurable where an open-ended task would read as noise.

It is also a different *kind* of task from both: TriviaQA is closed-book recall and GSM8K is
multi-step arithmetic, while MMLU is discrimination among given options. If the mechanism's
advantage were an artefact of free-form generation, it should not survive a forced choice.

## What is run

`analysis/build_bench_corpora.py::build_mmlu` writes `data/bench/mmlu_factual.jsonl` (500
questions, 5-shot, shuffle seed `0`) and the symlink data-dir `data/bench/mmlu`. Items come from
`analysis.selection_verifiable.load_mmlu` with a length filter at `2024` anchor tokens, so both
mechanisms answer the same questions from the same pipeline (caution (at)). The filter is a safety
net rather than a selection: the median 5-shot prompt is `360` tokens and `1999` of the first
`2000` items already fit in `1024`.

Then, exactly as `scripts/run_tqa_headtohead.sh` does for TriviaQA and with the same flags:

```
h1.py --k-values -1 0 0.5 1 3 20 --trajectories-per-prompt 1  --data-dir data/bench/mmlu ...
h1.py --k-values 0            --trajectories-per-prompt 64 --data-dir data/bench/mmlu ...
analysis/verifiable_metered.py --task mmlu --corpus data/bench/mmlu_factual.jsonl
```

anchor `TinyComma-1.8B`, risky `Llama-3.1-8B-Instruct`, `--max-new-tokens 24`, `--cap-factual 500`.
`k=-1` (risky alone) and `k=0` (anchor alone) are in the sweep, as every arm in this project
requires. Answers are graded by the first standalone `A`--`D` letter on the answer line, applied
identically to every arm.

## Bands, committed before the run

**H1 --- the entry gate. Can the anchor do this task at all?** Read on the `n=1` / `k=0` arm.

| reading | band |
|---|---|
| ANCHOR HAS SIGNAL | the `n=1` accuracy 95% interval **excludes** `0.25` |
| AT CHANCE | the interval contains `0.25` |

**If AT CHANCE, H2 and H3 are not read.** A mechanism that selects among chance-level draws cannot
be measured, and the arm is then reported as a third instance of the finding the paper already
carries --- where the anchor cannot do the task, no $n$ helps (`kl3m17bprobe`, `pleias3bprobe`).
**We think this gate is genuinely at risk:** TinyComma-1.8B is a 1.8B model trained on
public-domain and permissively licensed text, MMLU is hard, and `0.112` on TriviaQA is a low base.

**H2 --- does selection lift accuracy?**

| reading | band |
|---|---|
| SELECTION LIFTS | gain of majority vote at `n=64` over `n=1` is positive, interval excludes zero |
| FLAT | the interval contains zero |

**H3 --- the frontier, which is the claim.** For each accuracy selection reaches over the shared
control, take the cheapest metered arm reaching that accuracy and form
`selection nats / metered nats`. Selection's figure is the closed-form bound `log n - (n-1)/n`
and the meter's is its **measured** mean spend, so every ratio runs in the conservative direction
and the paper will say so.

| reading | band |
|---|---|
| **FRONTIER HOLDS** | every ratio is `< 1` |
| MIXED | some ratios `< 1` and some `>= 1` |
| FRONTIER FAILS | the median ratio is `>= 1` |

On TriviaQA the ratios are `0.032, 0.061, 0.093, 0.056, 0.071` at `n = 4,8,16,32,64`. **We predict
FRONTIER HOLDS and we predict the ratios will be larger than TriviaQA's**, because a forced choice
gives the meter a cheaper route to a correct token than free-form recall does.

## H4 --- the manuscript consequence, fixed now

- **H1 AT CHANCE.** No frontier claim from this arm. It is reported in one sentence in
  Appendix~I beside the two existing chanceless anchors, and the judge-free head-to-head remains a
  one-task result, **stated as such** in Limitations. The paper does not quietly keep saying "the
  judge-free axis" in the plural.
- **H1 passes, FRONTIER HOLDS.** Section 4.3's judge-free sentence names two tasks, and the
  Limitations sentence about a single judge-free head-to-head is struck.
- **H1 passes, MIXED or FRONTIER FAILS.** Reported as measured, in the main text, beside the
  TriviaQA result. A second task that disagrees with the first is the most informative outcome
  available here and it is not moved to an appendix.

## H5 --- what may not be claimed

No judged number, no certificate, no leakage and no `s(x)` number comes from this arm: MMLU answers
are not protected text and every cap here is `k*T_max` at `T_max = 24`. No accuracy level from this
arm is quoted beside a level from the TriviaQA arm --- different tasks, different floors. Only the
*ratios* are compared across the two tasks, and only as ratios.

## Excluded alternatives

- Changing `n_shot`, the length filter, the grading rule or the `k` grid after seeing any result.
- Dropping MMLU subjects, or reweighting them, to raise the anchor's accuracy past the H1 gate.
- Reporting H2 or H3 if H1 reads AT CHANCE.
- Substituting a different anchor. TinyComma is the only one a meter runs on, which is the whole
  reason this arm is at a weak anchor.

## Scoring log

## Scoring, 2026-09-20: the arm is INVALID, on two defects of our own

Run completed (`results/verifiable_metered_mmlu.csv`, generations in
`output/phase5/mmlu_{metered,sel64}`). **No band is read, and no number from this run enters the
paper.** Two defects, both in this document and in our code, not in the data.

### Defect 1 --- the grading rule cannot read the anchor's own output format

`extract_mmlu` took the first line of the completion and looked for a standalone `A`--`D`.
TinyComma-1.8B is a weak base model and **echoes the tail of the prompt before answering**:

```
prompt tail  ... C. Sioux Falls\nD. Pierre\nAnswer:
generation   ' Falls\nD. Pierre\nAnswer: D'
```

The first line is `" Falls"`, which carries no letter, so the item scored as wrong. This happened on
**`301` of `500`** anchor completions and `206` of `500` risky ones. The generations are fine; the
prompts are complete (verified by reading `prompt_text` end to end on a failing item, above); only
the parser is wrong.

**The tell was the contradiction, not the number.** A four-way choice has a floor of `0.25` under
random guessing. The anchor read `0.088` and `Llama-3.1-8B-Instruct` --- whose published MMLU is
around `0.68` --- read `0.276`. A model does not get *below chance* on a forced choice by being
weak; it gets there by not being read. Caution (t): a low number is the easiest kind of bug to
mistake for a result.

### Defect 2 --- the H1 gate has no direction, and would have passed this

The band reads *"ANCHOR HAS SIGNAL: the `n=1` accuracy 95% interval **excludes** `0.25`"*. The
measured interval is `[0.064, 0.114]`, which excludes `0.25` --- **downward**. Read literally the
gate passes and licenses H2 and H3 on a model performing at a third of chance, which is the
opposite of what the gate was for. The band should have said *above* `0.25`. It did not, and that
is recorded here rather than silently repaired.

### What happens now

Per caution (w), **a defect in our own specification must not retire the question**, so the arm is
INVALID rather than failed and MMLU is not dropped. Per this document's own excluded alternatives
--- *"changing ... the grading rule after seeing any result"* --- the rule may **not** be repaired
and the same numbers then reported. The repair and the re-score are therefore registered afresh in
`results/onset_prediction_mmlu_rescore.md`, with both bands corrected and the parser fixed and
mutation-tested **before** it is run on these generations (caution (ap): repair a gate before you
look at the result it gates).

What carries over untouched: the generations, which are deterministic given the seeds and were
produced before any of this was known, and the corpus, which is fixed in this document. What does
not carry over is every accuracy in `results/verifiable_metered_mmlu.csv`; that file is superseded
and is kept only as the record of this failure.
