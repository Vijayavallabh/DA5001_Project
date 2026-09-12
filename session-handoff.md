# Session handoff — 2026-09-12 (evening)

## Current objective

**feat-096 is the only thing in progress**: Comma-7B on AlpacaEval-805, the arm that decides between
the two open readings of the weaker standard-benchmark gain. Everything else through feat-095 is
`done` and verified.

The paper is v7. It argues one claim and exhibits a mechanism on the other side of it:

> One line of the chain rule leaves a per-token budget two options. Either `K = kT` grows with the
> work and the certificate is **vacuous**, or the budget is bounded and the decoder is the anchor at
> all but `O(1)` steps --- **trivial**. Both horns are measured. The escape is not a better meter
> but a different place to spend: `q(y) <= n p_s(y)` is a pathwise certificate of exactly `log n`,
> and it does not grow with the work.

**What changed today, in one line each.** The constructive claim stopped being a single setup
(B1 GENERALISES: four anchors, three families, and the strongest anchor gives the largest gain in
the paper). The support-ceiling *explanation* of the weaker benchmark gain was tested and does not
survive its own confound, so the paper says the gain is weaker off our prompts and that we cannot
say why. Figure 1 was illegible at print size and is redrawn.

## State

| | |
|---|---|
| manuscript | `~/sub/satml/iclr_2027.tex`, *Vacuous or Trivial* |
| build | `exit=0`, `overfull=0`, `unresolved=0`, main text **exactly 9 of 9 pages** |
| tests | **322**, all passing |
| numeric audit | 2,044 literals, 1 expected miss (`64256`) |
| compute | 149.6 GPU-hours measured, `150` disclosed |
| tree | clean, branch `iclr-2027` |

Sections: 1 intro · 2 theory (Props 1, 2, Thm 1) · 3 onset · 4 *The dichotomy's corollaries,
tested* · 5 selection anchoring (Prop 4) · 6 experiments · 7 related · 8 limitations and conclusion.

## What is running

| job | GPU | log | expected |
|---|---|---|---|
| feat-096 Comma-7B on AlpacaEval-805 | 2 | `output/logs/alpaca_comma7b.log` | ~4.5 h from 16:19; the first progress line appears only after 805 of 6,440 generations |
| B3 leakage, Comma-7B | 4 | `output/logs/b3_leakage.log` | ~45 min from 16:37 |
| B3 leakage, Pleias-1.2B | 0 | `output/logs/b3_pleias12b.log` | scoring phase |
| BookMIA regimes, seen + unseen | 0 | `output/logs/regimes_bookmia.log` | **done**, scored |
| Proposition 3 at two more pairs (`results/onset_prediction_imitation_breadth.md`) | 0 | `output/logs/imit_breadth.log` | ~2 h from 17:04 |

| feat-097 head-to-head at a second pair (`results/onset_prediction_frontier_second_pair.md`) | 4 | `output/logs/sel_llama321b.log`, `frontier_pair.log` | chained; scores as soon as generation lands |
| feat-098 Proposition 3 at TinyComma + Llama-3.1-70B (`results/onset_prediction_imitation_70b.md`) | 0+1 | `output/logs/imit_70b.log` | running since 19:43, ~3-4 h |
| feat-100 the strongest anchor at the largest n (`results/onset_prediction_selection_n64_comma7b.md`) | 4 | `output/logs/sel_comma7b_64.log` | running since 20:27, ~6 h generation |

The pre-registrations with no scoring section yet are
`results/onset_prediction_alpaca_comma7b.md`, `results/onset_prediction_imitation_70b.md` and
`results/onset_prediction_selection_n64_comma7b.md`; `tests/test_preregistration_count.py` fails if an
unscored one is not named here.

**The 70B arm takes two cards and waits for them.** It starts only when
`output/logs/.frontier_pair_done` exists *and* both GPU 0 and GPU 4 read under 2 GB, then
re-checks after a minute, so it cannot race the scoring pass onto a card someone else took. If it
prints "a card was taken while waiting" it exited without starting and can simply be relaunched.

## Recommended next step

1. **Score feat-096 against its committed bands.** `results/onset_prediction_alpaca_comma7b.md`
   fixes the manuscript consequence of each reading in advance: under PROMPT-SET EFFECT the paper
   leads with the AlpacaEval number in Section 6 **and** the abstract, whatever A1 reads. Scoring
   command is in the file.
2. **Finish B3 and append it to the breadth scoring log.** Two of three anchors are pending;
   `results/onset_prediction_selection_breadth.md` already carries B1 and B2 and says B3 is
   appended below when it lands. Any non-zero recall at any anchor goes in the main text whatever
   it does to the utility story.
3. **Run the CopyBench baseline for the BookMIA arm.** `analysis/regimes.py --model
   jacquelinehe/tinycomma-1.8b-llama3-tokenizer --data data --out results/regimes_copybench.csv`,
   then `analysis/bookmia_regimes.py --out results`. V2 needs the 16-novel median to compare
   against and it is the only input not yet computed.
4. Rebuild the artifact and refresh the compute figure once the queue drains.

## Still unscheduled from the scale audit

BookMIA-50 **onset** (needs memorisers re-fine-tuned, ~11 GPU-h for 3 pairs); Proposition 3 at two
more pairs (~14 GPU-h); the 70B comparison (~15 GPU-h on two cards). The BookMIA *vacuity* arm now
running covers the corpus-size objection for Proposition 1 only --- it says nothing about onset, and
the pre-registration names that as excluded alternative 5 so the two cannot be conflated.

## Traps this session added to AGENTS.md

- **(m)** the judge is position-dominated: the same two texts win 261/500 shown second and 24 shown
  first. Never quote an absolute judged level across passes; gains survive, levels do not.
- **(n)** the page budget moves with floats, headings and table rows, not with prose. A dozen
  sentence trims freed nothing; four structural changes freed five lines at once.
- **(o)** `h1.py` writes the CLI `--k-values` string verbatim into filenames, so `1e-9` becomes
  `trajectories_k1e-09_*.jsonl`.
- **(p)** a gate that fails everything is not a gate. The breadth entry gate read a column its
  producer never writes and had therefore never run; measure a gate against the artefacts it is
  about and assert that something passes.

And one that is not a trap but a habit: **render a manuscript page to PNG and look at it.** Figure 1
had been below any legibility floor for as long as it has existed, and no compile-time check saw it.
