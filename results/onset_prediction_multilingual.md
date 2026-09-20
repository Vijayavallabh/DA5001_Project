# Pre-registration: does any of this hold outside English prose?

Committed **before the memoriser is trained**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

Limitations says it plainly: *"Every extraction number is sixteen English novels of prose."* A
report flags it. The obstruction was never the pipeline --- `--corpus-file` and the symlink
data-dir construction already exist --- it was the corpus.

## The corpus, fixed here

`analysis/build_multilingual_corpus.py` builds it from Project Gutenberg. **These are
public-domain works used as stand-ins for protected ones**, exactly as the English second-corpus
arm does; nothing here is under copyright. What is measured is whether a model fine-tuned to
memorise them reproduces them, and whether the certificate holds when it does.

Composition, fixed now so a re-fetch cannot silently change it:

| split | passages | books | languages |
|---|---|---|---|
| `attack_train` | `128` | Der Prozess, Die Leiden des jungen Werther, Le Comte de Monte-Cristo, Notre-Dame de Paris | de, fr |
| `val` | `32` | the same four | de, fr |
| `test` | `80` | Don Quijote, Vingt mille lieues sous les mers | es, fr |

Splits are **disjoint in book** (caution (h)), asserted by the builder. Each passage is `930`
characters of seed and `250` of reference, matching the committed English corpora exactly, because
recall is otherwise confounded by how much context a corpus gives.

## The pairing, and why the anchor is not the audited one

Selection's guarantee is **relative to the anchor**, so running it with an anchor that does not
speak the language would make ``selection reproduces nothing'' true and uninformative --- there
would be nothing to reproduce. `Pleias-3B` is trained on Common Corpus, which is multilingual and
permissively licensed, so it satisfies the paper's own premise (trained without the protected
work) **and** can write French and German. The risky model is `Llama-3.1-8B-Instruct` fine-tuned
with LoRA on `attack_train` + `val` at the committed recipe defaults (rank `64`, `12` epochs,
`lr 2e-4`, stop-loss `0.03`, seed `0`). The scorer is `Qwen2.5-7B-Instruct`, as everywhere else.

## Bands, committed before the run

**H1 --- the admission gate, which is this project's standing rule (caution (a)).**

| reading | band |
|---|---|
| ADMISSIBLE | the memoriser's **sampled** `k=-1` near-verbatim recall on `attack_train` is `>= 0.10` |
| INADMISSIBLE | below `0.10` |

Greedy recall does not count: a 350M memoriser once scored `0.708` greedy and `0.022` sampled.
**If H1 fails nothing else is read** --- a certificate tested against an adversary that cannot
reproduce the work tests nothing. **We think this gate is at genuine risk**: the recipe was tuned
on English, and a LoRA of rank `64` may memorise non-English prose less readily.

**H2 --- the certificate's own claim, which is what the paper says.**

| reading | band |
|---|---|
| HOLDS | selection's near-verbatim recall is `0.0000` at every `n <= 64` |
| LEAKS | above `0.0000` at any `n <= 64` |

**H3 --- the premise, measured rather than assumed.** Selection's guarantee is relative to the
anchor, so the anchor's own leakage is the thing `n` multiplies.

| reading | band |
|---|---|
| PREMISE HOLDS | `Pleias-3B` alone (`n=1`) reads `0.0000` |
| PREMISE FAILS | above `0.0000` |

If H3 fails, H2 is **not** a success story and will not be reported as one: a clean selection arm
over a contaminated anchor is Proposition~4's failure case, not its confirmation.

**H4 --- the comparison with English, stated as a band so it cannot be narrated after the fact.**
On the English corpus at this scorer the memoriser reads `0.3925` sampled and selection `0.0000` at
every `n <= 64`.

| reading | band |
|---|---|
| SAME PICTURE | H1 ADMISSIBLE **and** H2 HOLDS **and** H3 PREMISE HOLDS |
| DIFFERENT | any of the three departs from its English counterpart |

## H5 --- the manuscript consequence, fixed now

- **SAME PICTURE.** The Limitations sentence ``every extraction number is sixteen English novels
  of prose'' is **replaced** by the measured scope: three languages, five further works. This is
  the only outcome that lets that sentence go.
- **H1 INADMISSIBLE.** The sentence stays, and gains the reason: we could not build an adversary
  in these languages at this recipe, so the question is open rather than answered. No H2 or H3
  number is reported.
- **H2 LEAKS.** Reported in the **main text**, not an appendix. A leak outside English is the most
  important negative result available to this paper and it is not buried.
- **H3 PREMISE FAILS.** Reported as a contaminated-anchor instance, with H2 explicitly **not**
  claimed as a success.

## What may not be claimed

No `\log n`, no vacuity threshold, no `s(x)` and no onset ratio comes from this arm --- those need
a per-token surprisal measured on the same anchor and corpus, which this arm does not produce. No
judged utility number: there is no judged workload in these languages.

## Excluded alternatives

- Changing the recipe (rank, epochs, learning rate, stop-loss) after seeing H1, or re-running with
  a different anchor after seeing H2.
- Dropping a language or a book after seeing per-book results.
- Reporting H2 while H1 is inadmissible, or while H3 fails, in either case as a success.
- Reading greedy recall where the band says sampled.

## Scoring log
