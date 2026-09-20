# Pre-registration: re-scoring the MMLU head-to-head under a repaired parser

Committed **before the repaired parser is run on any generation**. Nothing above the
`## Scoring log` line is edited afterwards.

## What happened, and what may be reused

`results/onset_prediction_mmlu_headtohead.md` records its arm INVALID on two defects of ours: a
grading rule that could not read the anchor's own output format (it took the completion's first
line, and TinyComma-1.8B echoes the prompt's tail before answering, so `301` of `500` anchor
completions scored as wrong and a four-way choice landed *below* its `0.25` floor), and an H1 gate
with no direction, which that sub-chance reading would have passed.

**Reused unchanged:** the generations in `output/phase5/mmlu_{metered,sel64}` and the corpus
`data/bench/mmlu_factual.jsonl`. Both are deterministic given seeds fixed in the earlier document
and both predate the discovery. Nothing is regenerated and no model runs.

**Not reused:** every accuracy in `results/verifiable_metered_mmlu.csv`. That file is superseded.

## The repaired rule, fixed and tested before it is run

`analysis/selection_verifiable.py::extract_mmlu` now reads the letter following the **last**
`Answer:` marker in the completion, cutting first at any run-on `Question:`; with no marker the
whole span is searched. The rule is derived from the few-shot **format** --- every shot ends
`Answer: <letter>` --- and not from which letter is correct, and it is applied identically to every
arm: anchor draws, selection, all six metered budgets and both `k=-1` baselines.

`tests/test_mmlu_extraction.py` pins it to the shapes actually present in the generation files, and
was **mutation-tested in four directions before the parser was run on them** (caution (ap)):
reverting to the first line fails it, taking the first `Answer:` marker rather than the last fails
it, dropping the run-on cut fails it, and the `\b`-boundary case that would read every completion
as `A` fails it. The first pass of that mutation test found the first/last choice **unguarded** and
a case was added for it.

## Bands, corrected and committed before the re-score

**H1 --- the entry gate, now with a direction.**

| reading | band |
|---|---|
| ANCHOR HAS SIGNAL | the `n=1` accuracy 95% interval lies **strictly above** `0.25` |
| AT CHANCE | the interval contains `0.25` |
| **BELOW CHANCE** | the interval lies strictly below `0.25` |

`BELOW CHANCE` is a **new reading and it is a failure of the instrument, not of the mechanism**: on
a four-way forced choice, sub-chance means answers are not being read, and if it recurs under the
repaired parser the arm is invalid again rather than negative. H2 and H3 are read only under
`ANCHOR HAS SIGNAL`.

**H1b --- an instrument check the first arm had no way to fail.** The unconstrained risky model
(`k=-1`, `Llama-3.1-8B-Instruct`) must read **at least `0.55`**. Its published 5-shot MMLU is around
`0.68`; a reading far below that means the parser or the prompt is still wrong, whatever the anchor
does. Under the broken parser it read `0.276`. **If H1b fails nothing else is read**, including H1.

**H2 --- does selection lift accuracy?** Unchanged from the superseded document.

| reading | band |
|---|---|
| SELECTION LIFTS | gain of majority vote at `n=64` over `n=1` positive, interval excludes zero |
| FLAT | the interval contains zero |

**H3 --- the frontier.** Unchanged: for each accuracy selection reaches over the shared control,
`selection nats / cheapest metered nats reaching it`. FRONTIER HOLDS if every ratio is `< 1`, MIXED
if some are `>= 1`, FRONTIER FAILS if the median is `>= 1`. Selection's figure is the closed-form
bound and the meter's is its measured mean spend, so the comparison runs in the conservative
direction and the paper will say so.

## H4 --- the manuscript consequence, fixed now

- **H1b fails, or H1 reads BELOW CHANCE.** The arm is invalid a second time, MMLU is reported in
  Limitations as a task we could not measure at this anchor, and **no MMLU number enters the
  paper**. We do not get a third attempt at this corpus: a rule repaired twice against the same
  generations is a rule fitted to them.
- **H1 AT CHANCE.** No frontier claim. One sentence in Appendix I beside the two chanceless
  anchors, and the judge-free head-to-head stays a one-task result, said plainly in Limitations.
- **H1 passes, FRONTIER HOLDS.** Section 4.3's judge-free sentence names two tasks and the
  Limitations sentence about a single judge-free head-to-head is struck.
- **H1 passes, MIXED or FRONTIER FAILS.** Reported in the main text beside the TriviaQA result. A
  second task that disagrees with the first is the most informative outcome here and is not moved
  to an appendix.

## What may not be claimed

No judged number, no certificate, no leakage, no `s(x)`. No accuracy level here is quoted beside
one from the TriviaQA arm --- different tasks, different floors --- only the ratios, and only as
ratios.

## Excluded alternatives

- Repairing the parser again after seeing this re-score.
- Changing `n_shot`, the corpus, the length filter or the `k` grid.
- Reporting H2 or H3 if H1b fails or H1 does not read ANCHOR HAS SIGNAL.
- Quoting anything from the superseded `results/verifiable_metered_mmlu.csv`.

## Scoring log

## Scoring, 2026-09-20

Re-scored with `analysis/verifiable_metered.py --task mmlu` on the same generations, repaired
parser, nothing regenerated.

### H1b --- the instrument check: PASSES

`Llama-3.1-8B-Instruct` at `k=-1` reads **`0.5860`** `[0.5420, 0.6300]`, against the registered
floor of `0.55`. Under the broken parser it read `0.276`. The parser now reads **`500/500`** of its
completions and **`443/500`** (`88.6%`) of the anchor's, against `194/400` and `172/400` before.
The instrument works.

### H1 --- the entry gate: BELOW CHANCE

The anchor at `n=1` / `k=0` reads **`0.1880`** `[0.1540, 0.2220]`. That interval lies **strictly
below** `0.25`, which is the registered `BELOW CHANCE` reading.

**So H2 and H3 are not read, and they were not computed.** The re-scored CSV contains selection
rows; they have not been opened, by the scorer or by hand, and
`tests/test_mmlu_not_quoted.py` fails if any of them ever reaches the manuscript. This is
feat-132's rule (caution (v)): when a gate fails, do not compute the band *just to see*, because
that is the only thing that keeps a later attempt honest rather than a second look at a number
already seen.

### The band's label is only half right here, and the consequence is honoured anyway

`BELOW CHANCE` was written as *"a failure of the instrument, not of the mechanism"*, on the
reasoning that sub-chance on a forced choice means answers are not being read. **That is only
partly the cause.** The parser now reads `88.6%` of anchor completions; if the remaining `11.4%`
were at chance the ceiling would be `0.2215`, and the measurement is `0.1880`. Among parsed items
accuracy is `0.212`, about two standard errors below `0.25`. The anchor also answers **`B` on
`217` of `443`** parsed items (`49%`) against the risky model's near-uniform
`A 102 / B 134 / C 101 / D 163`. The honest description is that **TinyComma-1.8B has no MMLU
ability and a strong position bias**, which is a property of the anchor and not only of our parser.

That reading would support the weaker, informative claim the paper already makes elsewhere --- that
where the anchor cannot do the task, no `n` rescues it. We do **not** make it from this arm. The
band was fixed before the re-score, it names this outcome, and its consequence is that no number
from this anchor's MMLU run enters the paper. Relabelling the outcome after seeing which half of
the cause dominates is exactly the move a pre-registration exists to prevent.

### Consequence, applied

- No MMLU-at-TinyComma number enters the manuscript, and none is quoted here beyond the two gate
  readings and the diagnostics above.
- Per H4, **there is no third attempt at this corpus at this anchor.** A rule repaired twice
  against the same generations is a rule fitted to them.
- The judge-free head-to-head against a metered decoder therefore remains a **one-task** result,
  TriviaQA, and Limitations says so rather than implying a plural axis.
- **The question is not retired.** `results/onset_prediction_mmlu_comma7b.md` is a separate arm at
  a capable anchor, registered before any of this was known and still running; it has its own H1
  gate and is untouched by this outcome. What this arm establishes is only that the *head-to-head*
  cannot be run on MMLU, because the one anchor a meter shares a vocabulary with cannot do MMLU.
