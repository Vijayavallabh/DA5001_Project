# Session handoff --- 2026-09-19 evening

## Current objective

The appendix reduction is **finished**: 46 -> 32 pages, appendix 33 -> **20** (pp. 13--32), main text
untouched at exactly 9 of 9 with the Ethics Statement at the top of page 10. 667 tests pass, exit 0,
0 overfull, 0 `??`. Nothing about the manuscript is outstanding except folding in the two GPU arms
below when they land.

## What was done

1. **Appendix 33 -> 20 pages, in four tranches** (commits `7367b16`, `5b96f53`, `f7bab87`,
   `7e062f1`). The rule throughout: **delete the apparatus, keep every concession, every guarded
   number and every claim the main text points at.** Retired whole: `appendix_seed` (kept verbatim as
   `sections/appendix_seed_v8_2026-09-19.tex`). Tables and figures cut where a proof, a table or the
   paragraph beside them already carried the content: the epoch/seed ladders, the scorer-scale grids,
   the vetting table, the two extra-pair head-to-heads, the TriviaQA certificate table,
   `tab:repairs`, the contamination table, five figures. The two-judge and two-pipeline paragraphs
   were **compressed, not dropped** --- they are concessions.
2. **The last three pages came from font size, not from deleting results.** The appendix body is now
   `\small` (floats are already `\footnotesize`, an absolute size, so no table or caption changed).
   That was worth 23 -> 20 pages on its own, against the alternative of deleting Proposition 4's only
   real test, the concession that only two anchors climb reproducibly, or the paper's own headline
   table. **Try the typographic lever before the scientific one** (AGENTS.md, caution (ar)).
3. **Read-through 6** found a caution-(aj) defect: two sentences, one in the main text, cited
   Appendix D for the ladders, which are in Appendix E. Re-pointed and guarded as a property.
4. `analysis/audit_numbers.py`: 2,188 numeric literals, **one** not in a CSV (`64256`, documented).

## Running now --- three cards, nothing to do but wait

| arm | class | card | progress at 15:10 | ETA |
|---|---|---|---|---|
| feat-134 | neutral 200x128 | GPU 1 | 200/25600 (relaunched 15:04) | ~28h |
| feat-134 | creative 150x128 | GPU 4 | 3900/19200 | ~16h |
| feat-134 | factual 150x128 | GPU 2 | 6450/19200 | ~14h |

* feat-134's neutral class had been **OOM-killed twice by another user's 51 GB process** (now gone).
  Its output directory was empty and `h1.py` truncates, so the relaunch is clean.
* `scripts/run_comma7b128_card2b.sh` (GPU 4) **owns the merge and the scoring**: it waits on
  `GEN_DONE` in the neutral and factual directories, both written only on `rc=0`, then merges and runs
  `selection_scaling.py --max-n 128 --tag _comma7b128`. Nothing to launch by hand.
* The old requeue waiter is **dead** (its log stopped at 11:32), so nothing will relaunch neutral
  underneath the running job.
* **feat-135 stage 1 (vetting) PASSED**: `results/vet_kl3m37b_base.csv` reads `0.0000` at every n and
  at `k=-1`, which is gate G1, so the anchor is admissible. Stage 2 is queued in
  `scripts/run_kl3m37b_breadth64_queued.sh`, waiting on the factual class's `GEN_DONE` to take GPU 2
  (~14h), then ~7 GPU-h. It runs `scripts/run_breadth64.sh` unmodified, so the protocol is identical
  to feat-130's three anchors by construction.

## Recommended next step

Check `output/logs/comma7b128_card2.log` for `COMMA-7B n=128 ARM DRAINED` and
`output/logs/breadth64_kl3m37b.log` for `kl3m37b DONE`. Then:

1. **Score feat-134** against the bands in `results/onset_prediction_comma7b_n128.md`. The sentence
   it lands in is in `sections/appendix_selection.tex`: "We did not take Comma-7B past $64$, so where
   the strongest anchor's ceiling sits is open."
2. **Score feat-135** against `results/onset_prediction_kl3m37b_breadth64.md`, whose verdict table is
   already written, including the **MARGINAL** rule at 2.0 interval half-widths. If it reads CLIMBS
   the appendix's two-anchor statement becomes three anchors in two families; if it saturates, the
   statement stays and gains the sentence that the family confound was tested.
3. Rebuild the artifact and re-run `analysis/compute_hours.py` once both arms close.

**Do not** compute either band before its gate passes (the rule both logs carry), and do not set a
single-order judged level from one sweep against another sweep's (caution (ap)).
