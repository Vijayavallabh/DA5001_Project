# Pre-registration: feat-179 Part A re-drawn under a disjoint seed (feat-182)

Committed **2026-09-23, before feat-179 Part A's bands B1--B3 are read** (8 of its 12 anchors had
landed; its scorer refuses to read them until all twelve have) **and before any draw of this arm
exists.** Nothing above the `## Scoring log` line is edited afterwards.

## Why

Part A (`results/onset_prediction_selector_n256.md`) replaces the published *"1.0 to 4.0"* --- a figure
in the abstract, the introduction, two main-text sections, a figure and an appendix --- with the
amplification the corrected selector realises over twelve contaminated anchors. It is one draw over
`100` passages, and its event is rare: `rate(1)` is a count of passages out of `100` from a single draw
per passage, and `A(64) = rate(64) / rate(1)` divides by it. One draw is exactly where this project has
watched a reading fail to survive a fresh one (caution (ap): KL3M-1.7B's paired difference moved
`0.061` under disjoint seeds). A number headed for the abstract should be one that survives a re-draw,
and a verdict should be stated only if it does.

## What runs

Part A's command character for character (`scripts/run_selfix_vetladder_local.sh`, `contam`), with
**one** change: `--seed 1234` becomes **`--seed 5678`**. The twelve anchors (`output/phase5/mem_*`),
the memoriser and selector (`output/memorizing_llama8b`), the `100` `attack_train` passages, `20`-token
seeds with the header, `200` new tokens, temperature `1.0`, `--n-values 1 8 64 256`, `--batch-size 32`
(batch size is part of the seed, caution (u), so it does not move) and `--experts-impl eager` where
Part A used it. Prefix `selfixR_<tag>`.

`--seed` seeds `torch` once per run, so the anchor's pool, the selector's picks and the memoriser's
`k = -1` baseline are all fresh draws. **The host does not change**: this arm runs on the local A100s
where Part A ran, so the seed is the only difference (caution (at): an arm that changes two things is
not the comparison it was registered to be). Each card's list waits for that card's last feat-179 or
feat-180 job (`scripts/run_selfix_redraw_local.sh`).

## Gates, read before any band

- **G0' --- the instrument, distributionally.** The `k = -1` draw is fresh, so Part A's bit-identity
  gate cannot apply. What must hold is that it is the same instrument: the same `100` `prompt_id`s as
  `contam_<tag>_per_passage.csv`, and the fraction of passages whose `k = -1` recall is at least `0.01`
  within a two-proportion `z` test (`|z| < 2.58`) of that arm's own, measured from its file by the
  scorer (caution (v)). This excludes gross defects --- a wrong split, a missing header, a wrong model,
  each of which moves that fraction by tens of points --- and nothing finer, which is a weakening and
  is said to be one (caution (as)).
- **G2** exactly as Part A's: served within the pool at every passage and `n`, the oracle
  non-decreasing, the pool counts consistent.
- Every anchor must land and pass both before any reading (the rule Part A's scorer now enforces).

## Bands

Part A's three readings --- B1 (SATURATES / GROWS, on `E_08`), B2 (TIGHT / LOOSE / BETWEEN / NO
READABLE ANCHOR) and B3 (STILL GROWING / SATURATED BY 64) --- are computed on this draw by the same
code (`analysis/selector_n256.py`) and set beside Part A's:

- **R1, R2, R3 --- REPLICATES** if this draw's verdict equals Part A's, **DOES NOT REPLICATE**
  otherwise.
- Per anchor, `A(64)`, `A(256)` and `T` on both draws side by side.

## What the manuscript does with each outcome, fixed now

- The amplification quoted at `n = 64` and `n = 256` is given **per draw**, both draws, never
  averaged into one rate.
- A verdict --- for instance that the adversary realises only a small fraction of the permitted
  amplification --- goes in the abstract or main text **only if it REPLICATES**. If it does not, the
  main text states it as draw-dependent and gives both readings.
- Part A's own readings stand as registered; this arm qualifies how they are stated and replaces none.

## Excluded in advance

- Changing anything but the seed; dropping an anchor; reading R1--R3 if G0' or G2 fails on any anchor
  (that is INVALID, and the anchor is re-run, not retired).
- Pooling the two draws, or choosing between them.

## What we predict

Part A has not been read, so we cannot predict the verdicts; we predict that **B1 and B3 replicate**,
because each turns on whether a count clears a wide threshold, and that **B2 is the one at risk**,
because `T` is a ratio of two small counts.

## Compute

Part A's anchors took `1.1` to `3.2` hours each on one A100 (its own logs), about `22` GPU-hours for
twelve, on the three local A100s behind the current queues: about seven hours.

## Scoring log
