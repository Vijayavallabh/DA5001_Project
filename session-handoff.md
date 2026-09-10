# Session handoff

## Current Objective

Plan v5, ICLR 2027 (abstract Sep 18, paper Sep 25). Branch `iclr-2027`; `master` holds the verified
SaTML paper at `dd7e801` and must not be deleted. The manuscript is `~/sub/satml/iclr_2027.tex`
(absolute `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml`), **not in this repo** -- never run
git after a `cd` into that tree.

State: main text **exactly 9 pages** (page 10 opens with the Ethics heading, which does not count),
0 overfull, 0 `??`, 1014 numeric literals audited with 1 expected miss (the independently verified
Comma-7B padded embedding count 64,256). **173 tests.** feat-035..069 `done`; nothing in progress.

## What landed this session

- feat-066 (done): all nine out-of-sample arms. Five seed arms, four temperature arms. The
  token-bucket rule beats a no-change null by 2.2x on the seed arms (5.1% against 11.0% mean
  relative error) and **loses** to it on the temperature arms (21.9% against 12.3%); the scorer and
  the paper report the two apart, because pooling would hide both.
- feat-068 (done): `analysis/matched_strength.py`, pre-registered in
  `results/onset_prediction_matched_strength.md`. Restricting both temperature arms to the passages
  the memoriser reproduces in **both** closes a strength gap of 0.519/0.904 to 0.947/0.993 and
  leaves the elasticity at +0.72 and +0.61, unchanged to two decimals. The warp acts through the
  anchor's rate, not the memoriser.
- feat-069 (done): Section 4 rewritten around three independent lines of evidence, main text held
  at 9 pages. `sections/onset_v2_2026-09-08.tex` is the previous version.

## Where the evidence for the units claim now stands

| line | what it is | strength |
|---|---|---|
| cross-pair, matched context | 5 pairs, ratios 0.878-0.926 over a 1.49x range of `s(x)`, cv 2.4%, leave-one-out 0.070 nats against a constant's 0.364 | observational; the grouping rule was read off the same seven measurements, so the two exact multiplicity checks (p = 0.048) do not test it |
| seed interventions | 5 arms, both directions, intervals disjoint from control on Arm B | causal within a pair, but non-monotone at 28 words and neither direction reaches the other family's band |
| within-pair warping | 2 arms, elasticity +0.72 and +0.61, both excluding 0 and 1 | the only design that moves `s(x)` itself; the one confound is controlled by feat-068 |

## Recommended next step

1. **Rerun `analysis/compute_hours.py`** once `output/phase5/{score_warp,warp_arms}.log` are more
   than 10 minutes old -- they were still inside the live-file window at the last run, which billed
   104.9 GPU-hours without them -- and update the figure in the LLM-usage statement
   (`iclr_2027.tex`, currently says 93).
2. Rebuild the artifact (`scripts/build_artifact.sh artifact`) so it carries
   `analysis/matched_strength.py`, `results/matched_strength.csv` and the new pre-registration, and
   add the Phase 5d reproduction command to `README_artifact.md`.
3. A full read-through of the manuscript end to end. Section 4, the introduction, the abstract, the
   conclusion and three appendices all changed today and have only been checked number by number.

## Cautions that cost time this session

- **Check tectonic's exit status, not just its overfull count.** A missing figure halts the build
  and leaves the *previous* PDF in place; grepping that stale PDF reported 9 pages and 21 total when
  the real document was 28 pages and over the limit. Always read `err=` and the page count together.
- The page budget had **already** been exceeded before this session's edits; the handoff's "exactly
  9 pages" was stale. Verify it against a fresh build, not against the last note.
- A bootstrapped threshold crossing needs its **no-crossing fraction** reported. Twice now a grid
  whose top end was a ceiling produced a narrow interval that was narrow precisely because it was
  conditioned on the resamples that happened to cross.
- `pgrep`/`pkill -f <pattern>` matches the invoking shell. Kill by PID and confirm with `kill -0`.
