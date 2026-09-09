# Session handoff

## Current Objective
Plan v5, branch `iclr-2027`, target ICLR 2027 (abstract Sep 18, paper Sep 25). Today (2026-09-10)
finished plan item D (the judged arms) and opened feat-064, which looks like the strongest new
result in the paper.

## What landed today

**1. The judged arms, at 600 pairs per arm and two judges (item D, complete).**
The published headline was under-powered and is now replaced. At 180 judged pairs the k=1
separation from the anchor reads -0.45 sigma; at 600 it reads -3.66. The old null arm (41.7% loss)
was the outlier against two runs that agree at 46.4/47.0%.

    k      vacuous%   Qwen z   Phi z    verdict
    0.5         0.0    -1.03   +1.52    neither judge separates the decoder from the anchor
    1-5     44-100     -3.66..  +0.11.. judges disagree
    10, 20    100.0    -6.97    -3.29   BOTH separate

The claim that survives both judges: **the useful region opens only where the certified one has
closed**. The judges disagree about the crossing by nearly an order of magnitude (0.68 vs 5.39),
reported as a result. alpha=4 fills Table 2's blank cell: 90% of steps overridden, 24x less oracle
leakage than alpha=1, and neither judge can separate them -- activity is not price.
Scripts: `analysis/judge_separation.py` (reproduces the published -0.45/-2.35 exactly on the old
input), `analysis/utility.py --prefix`, `analysis/utility_price.py` (multi-directory).

**2. feat-064: the onset split looks like the attack's seed, not the tokenizer.**
`composition_attack.py` seeds a fixed number of *tokens*, so KL3M's adversary holds 7.3 words and
the other five hold 13.0-14.4 -- an exact split with no overlap, matching the onset split.
Pre-registered in `results/onset_prediction_seed.md` (commits `bc74f4d`, `9ea3bec`, `a63ec8f`), then:

    Pleias-1.2B   seed words   onset   ratio   95% CI          (Arm B)
    seed 20             13.7   2.780   0.866   [0.72, 0.95]
    seed 10              6.9   3.272   1.004   [0.99, 1.23]    disjoint

Observationally the trend runs through all seven pairs (Spearman -0.919, exact permutation
p = 0.0071) and continues inside the coarse family alone (-0.800 over a 1.4-word range), though that
is confounded with granularity by construction -- only the intervention separates them.
Grounded in the literature and verified verbatim against the PDF: Carlini et al.
(`carlini2023quantifying`, already in the bib) define extractability "with k tokens of context",
state the conversion for one tokenizer ("Fifty tokens corresponds to an average of 127 characters or
25 words"), and report 33% -> 65% extraction from 50 -> 450 tokens of context.

**3. A false alarm, caught and reverted the same day.** I wrongly marked feat-060 invalid. See the
correction in `progress.md`: recall is scored against the decoded `target`, not the CopyBench
`reference` field, and `prompt_text` is 930 characters of the same novel, so a truncated target is
still protected text. Its k=-1 baseline is 0.696. feat-060 stands.

**4. `analysis/audit_numbers.py`**, new: every numeric literal in math mode checked against every
value in `results/**.csv`. Its first run found a stale Theorem 1 pricing table in `frontier.tex`.
615 -> 621 literals, 1 unsourced, and that one verified against the checkpoint config.

## State
- 165 tests; `./init.sh` passes. feat-035..063 done, feat-064 in progress.
- Manuscript: 19 pages, main text exactly 9, 0 overfull, 0 `??`, snapshot refreshed in
  `manuscript_snapshot/`.

## Running when this was written (all under `output/phase5/`)
- **Arm A**, `seed40_kl3m520m` on GPU 1: KL3M-520M at `--seed-tokens 40` (80.8 chars, 14.3 words,
  matching TinyComma's 81.3/14.4). Past k=2.2 with recall 0.004, so its onset is above 2.2.
  **This is the decisive run**: `results/onset_prediction_seed.md` commits two accounts that predict
  different answers -- seed-matching says 0.85-0.95, the k_crit account says 0.96 -- and an outcome
  in 0.93-0.95 is to be reported as undecided.
- **`seed_queue.sh`** behind it on GPU 1: KL3M-1.7B at seed 40 (the second fine pair, which with
  Arm A decides whether the seven-pair table collapses to one band), then the dose-response arms at
  seeds 10 and 80.
- **`util_cross`** on GPU 2: generation at k in {0.6, 0.7, 0.8} to pin the utility crossover,
  pre-registered in `results/onset_prediction_crossover.md`; both judge chains armed behind it.
- Score chains armed: `score_seed.sh`, `judge_cross.sh` x2.

## Recommended next step
Score Arm A with `analysis/seed_effect.py` and write Section 4 around whichever account survives.
If Arm A and the KL3M-1.7B arm both land in the coarse band, the seven-pair table collapses under a
character-matched protocol and the paper gains a law plus a protocol correction; if they land near
0.96, the mechanism is `k_crit` and the seed acts through where the target starts.

## Open, logged, not fixed
- Checkpoint-trust posture is inconsistent (`a_patch/factory.py` refuses pickled checkpoints; four
  analysis scripts do not). A security decision for the user.
- `results/onset_ladder.md` is stale: it predates pairs 4-7.
- A second protected corpus is not cheaply available; see the decision note in `progress.md`.

## Process notes that have each cost real time
- **`pgrep -f <pattern>` matches the shell running it.** Three incidents, the last a `pkill` that
  killed the invoking shell mid-heredoc and silently dropped a file. Wait on a PID with `kill -0`.
- **Before calling a committed run invalid, read what the metric compares against, in the code, and
  check the run's own k=-1 baseline.** A baseline near the ceiling means the measurement is real.
- **A shared scorer must not hardcode its output filename.** Reusing `score_truncation.py` silently
  overwrote `results/truncation_score.csv`; it now takes `--out-name`.
- **Chain scripts need `set -e`.** One printed `=== DONE ===` after an OOM.
