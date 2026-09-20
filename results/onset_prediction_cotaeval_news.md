# feat-155 --- Does the judge-free result hold on the community-standard benchmark, in a domain this paper has never used?

Everything above `## Scoring log` was written and committed **before any generation**, and nothing
above that line is edited afterwards.

## Why

The Program Chairs' report asks for the method to be benchmarked on **CoTaEval**
\citep{wei2024cotaeval} "rather than relying solely on the customized CopyBench and BookMIA
setups". That is the one point of theirs this paper still answers only partially: CoTaEval is
cited, and the MemFree rule it benchmarks was measured as a decoder, but no number in the paper
comes from CoTaEval itself.

It is also a **domain** gap and not only a framework one. Every protected corpus here is books ---
CopyBench (16 novels), BookMIA (100), Gutenberg, and the French/German multilingual arm. CoTaEval's
other half is **news**. This arm tests framework and domain at once.

## What is run

CoTaEval's news files, taken as distributed and not re-normalised (NewsQA is PTB-tokenised, with
`-LRB-` and spaced punctuation; normalising the corpus would make our numbers incomparable with
CoTaEval's own):

| file | n | role |
|---|---|---|
| `newsqa_indomain_utility.json` | `500` | **primary**: article + question + `Answer:` |
| `newsqa_blocklisted_infringement.json` | `1000` | secondary: `prompt_autocomplete` -> `gt_autocomplete` |

Anchor **Comma-7B** (`common-pile/comma-v0.1-2t`), the anchor the judge-free GSM8K and TriviaQA
arms use; pointwise reward `Qwen2.5-7B-Instruct`, the same fixed template; `n` in
`{1,2,4,8,16,32,64}`; seeds `42 43 44`. Mandatory baselines per the project rule: **`k=-1`** (risky
model alone) and **`k=0`** (anchor alone, which is `n=1`).

**Metric: SQuAD token-F1**, which is CoTaEval's in-domain utility metric and is graded. Exact match
is reported beside it and is **not** the primary, for a reason fixed before any band: six anchor
generations were read first (caution (au)) and they show run-ons, trailing punctuation, empty
strings, and --- decisively --- **partial answers** (`' On the morning of May 25'` against a gold of
`'May 25 , 1979'`). Exact match scores that zero. Registering exact match here would have repeated
the defect that invalidated the MMLU and LAMBADA arms.

`extract_cotaeval` cuts at the first newline and strips trailing punctuation. It was written
against those six observed generations, not against hypothesised shapes.

## Instrument gates, read FIRST and before any band

* **G1 (the task is doable at this anchor).** The anchor's `n=1` F1 must be **`>= 0.10`**. Below
  that the anchor cannot read the article at all and no selection number over it means anything ---
  this is the MMLU failure's signature (a model at a third of chance) and it stops the arm.
* **G2 (the stronger model is better).** The unconstrained risky model's F1 must exceed the
  anchor's `n=1` F1. If it does not, the pipeline is wrong, not the mechanism, and the arm is
  **INVALID** rather than a result.
* **G3 (non-degenerate).** The `n=1` empty-answer fraction is **reported, not gated** --- there is
  no prior for this corpus (caution (v) --- feat-132 failed by gating a rate against a sibling arm).
  Above `5%` it is recorded, not exempted, and the gain is also reported on non-empty prompts.
* **G4 (coverage).** All `500` items at every `n`.

## The committed band --- paired `F1(64) - F1(1)`, within this pass

Paired over the same `500` items and bootstrapped **within one pass**, which is what makes it
immune to grid-dependence (caution (ap)).

| reading | verdict |
|---|---|
| `> 0`, interval excludes `0` | **HOLDS ON COTAEVAL.** The judge-free result is not an artefact of our own corpora or of the book domain, and the paper may say the mechanism was benchmarked on the standard framework. |
| interval contains `0` | **NO EFFECT ON COTAEVAL.** Reported as a limit: the judge-free climb is visible on GSM8K and TriviaQA and not on CoTaEval news, and the paper says so. |
| `< 0`, interval excludes `0` | **TURNS OVER.** Reported in the main text; a selection arm that loses utility with `n` on a standard benchmark would qualify the headline. |

**Marginality, declared in advance.** Below **`2.0`** interval half-widths this is recorded MARGINAL
and is not promoted without a seed replication, exactly as feat-135 was (caution (ap)).

**We predict HOLDS.** GSM8K lifts `0.320 -> 0.546` at `n=32` and TriviaQA majority vote
`+0.054 [+0.026, +0.082]` at `n=64`; if the mechanism is real and not corpus-specific, a reading
comprehension task with a graded metric should climb too.

## The secondary, and what it may NOT be called

The infringement half is run and reported, and it is a **negative control, not CoTaEval's
memorization setting**. CoTaEval's memorization scenario fine-tunes a model on the corpus; neither
our anchor nor our risky model has ever seen NewsQA, so near-zero similarity there is the expected
and uninformative outcome, and it will be reported as a corpus-level negative control in a new
domain --- never as evidence that the mechanism suppresses memorised news. Claiming otherwise would
be caution (h)'s defect: scoring "protected" text a model never saw.

Reported: near-verbatim recall, ROUGE-L, LCS-word at every `n` plus `k=-1` and `k=0`.

## Excluded in advance

* Reading any `n > 1` number if G1 or G2 fails.
* Presenting the infringement half as a memorization result.
* Comparing any F1 level here against a level from another sweep (caution (ap)).
* Substituting exact match for F1 after seeing which is kinder.

## Compute

`500 x 64` plus `1000 x 64` at `7`B on idle H100s. Comparable arms took `2.5`--`3.5` h per `32,000`
trajectories, so this is **`~13` gpu-hours**, under the `24`-gpu-hour escalation threshold, on host
B at the user's instruction to use it.

## Scoring log
