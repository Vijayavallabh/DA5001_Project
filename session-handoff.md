# Session handoff — 2026-09-16 08:25 (EVERYTHING CLOSED; no GPU work in flight)

## Current objective

Everything the previous handoff queued is **done and committed**. Five commits since `3f047e8`,
each one scoring a band or repairing a claim the scoring exposed. One sweep is still running on
GPU 2; it cannot change any verdict and is wanted only for a committed secondary.

**51 pre-registrations, all scored** (`tests/test_preregistration_count.py` agrees: zero unscored).
**524 tests. `./init.sh` exit 0. Manuscript compiles exit 0, 0
overfull, 0 `??`, 9 of 9 body pages (0 body lines on page 10), 58 total, 3321 numeric literals with
the one expected miss. Artifact 886 files. 261.7 GPU-hours measured.**

---

## READ THIS FIRST: `nvidia-smi` is broken, and the cause is in our own repo

```
nvidia-smi                          -> Failed to initialize NVML: Driver/library version mismatch
env -u LD_LIBRARY_PATH nvidia-smi   -> prints the table
```

`LD_LIBRARY_PATH` leads with `NVIDIA-Linux-x86_64-580.173.02/` — 1.5 GB extracted in the repo root,
gitignored, **not created by this work** — whose `libnvidia-ml.so.1` shadows the system's and does
not match the loaded kernel module (`580.178.04`).

**CUDA compute is unaffected.** Every measured number is correct; the only casualty is NVML, which
is what torch calls inside `generate()` — that is why four fine-tunes died at 05:00 in their
*post-training* check, after writing their merged model, and each queue shell then skipped the sweep
behind it.

`scripts/gpu_env.sh` strips it and is sourced by four of the five queue launchers.
**`scripts/run_strength_ladder.sh` is still unpatched** — it was executing when the fix landed, and
bash reads a script incrementally by byte offset, so editing a running one corrupts it. Patch it
once its queue exits; the line is one `.` and can be copied from any of the other four. Do not
delete the directory: it is not ours. Caution (ab) in AGENTS.md.

---

## What is running

**Nothing.** The seed arm finished at 08:18 and its queue shell has exited; no job of ours holds a
GPU. The other-user job on GPU 0 (63.7 GiB) is untouched. The standing rule is back in force with
nothing to apply it to: **all future processes on GPU 2**, quoted inside `run_strength_ladder.sh`
and `run_copybench_seeds.sh`, both defaulting `GPU` to 2.

All three items the previous handoff queued are done:

1. **Seed 3 scored.** Sampled `k=-1` `0.2391`, ratio `0.9677` — the strongest of the four
   memorisers carrying the lowest ratio. The committed four-point span is **`0.3469`** against a
   committed `0.20`: unchanged in kind, stronger in degree, as the monotone argument predicted.
2. **The committed secondary scored.** Seven points, **`rho = -0.714`, exact `p = 0.0881`** (floor
   1/2520). Reported as registered: it does **not** reach significance. Direction consistent with
   every other ladder, magnitude unresolved.
3. **`scripts/run_strength_ladder.sh` patched** to source `scripts/gpu_env.sh`, now its queue has
   exited. All five launchers are guarded by `tests/test_bookmia_onset.py`.

### A protocol incident, found while checking that queue's exit

`run_strength_ladder.sh` was edited and committed at **03:17 while the queue started at 02:57 was
executing it**. Bash reads a script incrementally by byte offset, so the edit shifted every later
offset. The loop body had already been parsed and all three iterations ran the original code, but
on returning to the file after the loop the shell landed mid-command — `line 52: --base: command not
found`, then a spurious `FAILED finetune seeds=3`.

**No measurement is affected**, and that was checked rather than reasoned: all four points carry the
identical 16-point grid, `n=100`, `k=0` exactly `0.000`, zero invariant violations, and identical
`lr`, `rank`, `batch`, `accum`, `stop-loss` and base model in their recipes. Recorded in the scoring
log. **Never edit a shell script while it is running** — the failure surfaces far from its cause and
looks like a job failure rather than a source edit.

---

## What the session established, and what it cost us

### Four seed ladders. The fourth overturned a claim written six hours earlier

| ladder | memoriser `k=-1` | stop-loss fired? | strength span | **ratio span** |
|---|---|---|---|---|
| KL3M-520M, CopyBench | 0.5149 – 0.5756 | yes, 11/40 every seed | 1.12x | 0.0333 |
| KL3M-520M, BookMIA | 0.6690 – 0.7326 | yes, 26/40 every seed | 1.10x | 0.0663 |
| Pleias-1.2B, CopyBench | 0.9091 – 0.9615 | yes, 29–31/40 | 1.06x | 0.0795 |
| **Pleias-1.2B, BookMIA** | **0.1504 – 0.2045** | **no — 40/40 every seed** | **1.36x** | **0.2597** |

At 06:35, on three ladders, the manuscript was given *"Seeds do not move the onset ratio; training
length does."* The fourth refutes that as stated. What all four support, and what the paper now says:

> The onset ratio is reproducible under a re-seeded recipe **wherever the fine-tune converged and
> the memoriser is strong** (0.033–0.080), and **not** where it is marginal (0.2597). In neither
> case is it a property of the pair alone.

Bands: `seedspread.md` **RUN-TO-RUN VARIATION** (0.2597 ≥ the committed 0.20);
`seedspread2.md` **INCONCLUSIVE** (0.0663/0.2597 = 0.2553, missing the "below a quarter" band by
`0.0014` — the band was not moved).

### Three manuscript claims were wrong and are now right

1. **`appendix_seed.tex` (07:21).** The `>10`-word family disperses *less* across five different
   pairs (sd `0.0212`) than Pleias-1.2B, a member of it, does across three re-seeded fine-tunes (sd
   `0.0450`). Its tightness — and the leave-one-out `0.070` nats and exact `p = 0.008` resting on
   it — is **not resolved** against fine-tune noise. The family *separation* survives and is now
   said to. The matched-context `move` column is paired within one memoriser so the noise cancels;
   the `spread` row is not, so its residual `0.113` is an upper bound.
2. **The strength column (07:36) caught a wrong number on its first run.** Two appendices said the
   nine memorisers span "a factor of 3.4, from 0.2696 to 0.9091". **`0.2696` is `fineg_phi35` on
   Gutenberg**, not a CopyBench pair. True range `0.1806`–`0.9236`, **factor 5.11**.
3. **The seed-word gradient (08:02) is quantified, not hedged.** Under the largest measured re-seed
   noise it keeps its sign in **100%** of 20,000 draws and `p<0.05` in **92.3%**, but its median
   falls to **-0.849** with a 5–95% range of `[-0.958, -0.647]`. The direction is not at risk and
   the magnitude is; `-0.958` is the top of that range, not its centre.

### Two defects in our own data that the work surfaced

- **`output/phase4/fine_tc` and `fine_comma` ship no `k=-1` or `k=0` arm**, against Working Rules'
  mandatory-baselines requirement. Their baselines are borrowed from companion runs and daggered in
  the table, and only after `analysis/onset_table.py` asserts the companion's `[ca]` protocol line
  is identical to the sweep's, character for character. `--strict` refuses the borrow.
- **A strength must be read off the same passages as the onset beside it.** Pleias-350M is measured
  at n=100 and n=458 and the table prints the 458 row, so its strength is `0.875` and not the
  manifest sweep's `0.906`. The script now refuses any mismatch.

### Two things checked rather than assumed, both of which survived

- **Convergence does not explain the nine-pair ordering.** Four of nine reached their stop-loss,
  five did not, the groups interleave in rank, and the converged four still span `0.1748`.
- **Memoriser strength does not rank the nine.** Spearman `-0.483`, exact `p = 0.1938` over all
  362,880 permutations — so "strength fails to explain the ordering" still holds. The sign is the
  one every within-pair ladder shows, and the appendix now states the number rather than leaving a
  reader with the new column to discover it.

---

## Commits this session

| commit | what |
|---|---|
| `ea3bce1` 07:11 | SCORED the last two seed arms; the fourth ladder overturns the three-ladder claim; NVML root cause + `scripts/gpu_env.sh` |
| `8c381fe` 07:21 | `appendix_seed.tex`: the tight subgroup is tighter than its own member's retraining noise |
| `a2935ba` 07:36 | memoriser-strength column; caught the Gutenberg number in a CopyBench claim |
| `7a50701` 07:54 | convergence marker (`ep.` column) and the passage-set invariant it forced |
| `4f2a02a` 08:02 | seed-word gradient qualified by measurement |
| `ebb5d65` 08:10 | handoff rewrite; five estimated timestamps repointed to their commit times |
| `d92e162` 08:12 | both seed logs re-headed under the project's `## Scoring, <date>` convention |

**One defect the handoff rewrite exposed.** `tests/test_preregistration_count.py` had been passing
for the wrong reason: it flags an unscored pre-registration that the handoff does not name, and the
previous handoff named both seed arms as in flight. They were in fact scored at 07:11 — but under a
heading of my own spelling (`### SCORED ...`) rather than the project's `## Scoring, <date>`, so no
check could see it. Both logs now use the convention; the count is machine-verified at zero
unscored. A scoring log that a tool cannot recognise as scored is not scored.

New analyses: `analysis/onset_group_dispersion.py`, `analysis/seedword_gradient.py`.
New tests: `test_onset_group_dispersion.py` (7), `test_seedword_gradient.py` (6); `test_onset_table.py`
2 -> 9; `test_bookmia_onset.py` +6. Every one demonstrated to fail on its reintroduced defect.

---

## If a next session wants work beyond finishing the sweep

The queue the previous handoff carried is empty. Candidates, none of them blocking:

1. **`output/phase4/fine_tc` and `fine_comma` still lack their own baselines.** The decode is ~80 s
   each. Running them would retire the two daggers and bring both sweeps into compliance with the
   mandatory-baselines rule rather than working around it.
2. **A full read-through.** Three claims were wrong this session and all three were found by
   building a column or scoring a band, not by reading. Nothing has read the assembled document
   end to end since the v8 reorder.
3. **`fineb_kl3m_s3`/`s4`** memorisers are on disk with their sweeps deliberately dropped as
   secondary-only by their own pre-registration. Their stop epochs (26/40, matching seeds 0–2) are
   already used as the committed diagnostic.

---

## Standing constraints, unchanged

Never push to a remote. `feat-016` is human-only and must never be started. Do not modify
`~/sub/neurips_2026.tex`, `output.zip`, or the committed prompt sets under `data/`. Never GPU 3
(4 GB T400); always `CUDA_DEVICE_ORDER=PCI_BUS_ID` with `CUDA_VISIBLE_DEVICES`. Ask before
fine-tuning above 8B, any run over 24 GPU-hours, deleting a file the agent did not create, or
anything touching a remote. `pgrep -f`/`pkill -f` match the invoking shell — kill by PID. A budget
violation is per-trajectory, never a mean-bound artefact. Baselines at `k = -1` and `k = 0` are
mandatory. The manuscript lives in `~/sub/satml/`, **outside this repo, inside a stray home git repo
that must never be committed to**; after any edit there: recompile, check exit status, 0 `??`, 0
overfull, and count body lines on `pdftotext` page 10 (must be zero). Render a page to PNG and look
at it — it caught two defects today that no compile-time check sees.
