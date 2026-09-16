# Session handoff — 2026-09-16 11:35 (a crossover in flight on four cards; window to 18:36)

## Current objective

Every item earlier handoffs queued is closed. The GPUs are now spending a **bounded multi-GPU
window** on a crossover that tests a claim this session put into the paper and labelled post hoc.

**55 pre-registrations, 53 scored, 2 in flight (named below). 536 tests. `./init.sh` exit 0.
Manuscript compiles exit 0, 0 overfull, 0 `??`, 9 of 9 body pages (0 body lines on page 10), 58
total. Artifact 887 files. Compute 266.2 GPU-hours.**

---

## READ FIRST: `nvidia-smi` is broken, and the cause is in our own repo

```
nvidia-smi                          -> Failed to initialize NVML: Driver/library version mismatch
env -u LD_LIBRARY_PATH nvidia-smi   -> prints the table
```

`LD_LIBRARY_PATH` leads with `NVIDIA-Linux-x86_64-580.173.02/` — 1.5 GB extracted in the repo root,
gitignored, **not created by this work** — whose `libnvidia-ml.so.1` shadows the system's and does
not match the loaded kernel module (`580.178.04`). **CUDA compute is unaffected**; NVML is the only
casualty, and NVML is what torch calls inside `generate()`, which is why four fine-tunes died in
their post-training check on 2026-09-16 *after* writing their merged models.

`scripts/gpu_env.sh` strips it and **all five original launchers plus the four new ones source it**.
Do not delete the directory: it is not ours. Caution (ab) in AGENTS.md.

---

## The multi-GPU window

Opened 10:36 ("for the next 8 hours, use all the gpus"), **expires 18:36 today**. It SUSPENDS, not
cancels, the one-card rule. Every launcher takes `GPU` as an override **defaulting to 2**, so when
the window closes the standing rule restores itself with no action. GPU 3 is still never used, and
`CUDA_DEVICE_ORDER=PCI_BUS_ID` is still mandatory.

## What is running

| cards | job | started | expected |
|---|---|---|---|
| 0, 1, 4 | forward probes **at 60 epochs**, rates 1.5e-4 / 1e-4 / 5e-5 | 11:31 | ~12:46 |
| 2 | reverse probes, rates 6e-4 / 1e-3 / 2e-3 (co-located) | 10:46 | ~12:39 |

A background task waits on both families. Each card holds ~12.5 GB of 80 GB.

### The crossover, and the two unscored pre-registrations

The appendices say, as of this morning, that the onset ratio is reproducible **wherever the
memorisation fine-tune converged** — and label it post hoc, because the one irreproducible cell is
also the only non-converged one *and* the only marginal memoriser. Three things move together.
Convergence here is manipulable, so it can be pushed in both directions on different pairs:

| arm | pre-registration | cell | intervention | span to beat |
|---|---|---|---|---|
| forward | `onset_prediction_convergence_causal_60.md` | Pleias-1.2B BookMIA, never converges | lower the rate until the stop-loss fires | `0.2597` → below `0.10`? |
| reverse | `onset_prediction_convergence_reverse.md` | KL3M-520M BookMIA, converges 26/40 every seed | raise it until it stops firing | `0.0663` → above `0.20`? |

Both fix the rate-selection rule before any result exists, carry written invalidity conditions, and
commit to reporting the **strength band** — a rate change plausibly moves memoriser strength as well
as stability, and strength is what the ratio tracks (`rho = -0.714` over seven pooled points). That
is the confound the crossover exists to break, and it is disclosed in both files rather than found
later.

### Stage 1 of the forward arm came back INVALID, and that is recorded

`onset_prediction_convergence_causal.md` (the 40-epoch original) is **scored INVALID**: none of the
three rates crossed the `0.02` stop-loss (`0.0229`, `0.0260`, `0.0305`), so by its own written
condition the intervention was not built, and **its stage 2 was not run**.

The rates were not the problem. At `lr 1.5e-4` the last four epochs read `0.0269`, `0.0254`,
`0.0236`, `0.0229` — monotone and still descending — against the published `lr 3e-4`'s non-monotone
`0.0623 / 0.0865 / 0.0258 / 0.1098`. The oscillation the intervention targeted is gone; the epoch
cap stopped it short. Hence the 60-epoch retry.

**Why that is not a second bite at the apple, and the test to apply if this recurs:** no ratio had
been measured. Nothing was swept, no onset existed, stage 1 produced four loss numbers. Re-attempting
a construction that failed to construct is a different act from re-running an experiment whose answer
one dislikes. **Had one seed been swept, the honest course would have been to stop**, and both files
say so. The retry also commits to *abandonment* rather than a third probe if 60 epochs also fails.

### When the probes land — exactly this, in order

```bash
.venv/bin/python /mnt/md0/select_rate.py     # applies each arm's COMMITTED selection rule
#   fwd: the LARGEST rate that converges.  rev: the SMALLEST that does not.
#   Check both invalidity conditions before launching anything.

# then, per arm, one queue shell per seed (seed 0 reuses the probe's memoriser and sweeps only):
bash scripts/run_convergence_stage2.sh fwd <lr> <seed> <gpu>
bash scripts/run_convergence_stage2.sh rev <lr> <seed> <gpu>
```

Stage 2 is ~50–75 min of fine-tune plus ~3.5 h of sweep per seed; three seeds per arm in parallel
fit the window if they start by ~13:00. **If they cannot finish before 18:36, start them anyway on
GPU 2 and let them run** — the window governs how many cards may be used, not whether work may
continue.

---

## Closed this session (11 commits since `3f047e8`)

- **Four seed ladders scored.** The fourth overturned "seeds do not move the onset ratio", written
  six hours earlier; the paper now says reproducible *where the fine-tune converged and the
  memoriser is strong* (0.033–0.080) and not where it is marginal (0.2597 / 0.3469).
- **Three manuscript claims corrected:** `appendix_seed.tex`'s tight subgroup is tighter than its own
  member's retraining noise (sd 0.0212 over five pairs vs 0.0450 over three re-seeds); the strength
  range was quoting a **Gutenberg** number in a CopyBench claim (0.2696 → true 0.1806–0.9236, factor
  3.4 → 5.11); the seed-word gradient is qualified by measurement (sign survives 100% of 20,000
  redraws, median falls to −0.849).
- **Table 2 gained a `mem.` strength column and an `ep.` convergence column.**
- **All six genuine mandatory-baseline gaps closed:** `fine_tc`, `fine_comma` and the four Rényi
  orders. Rényi scored **CONFIRMED** — 16 arms, 16 exact reproductions of `fine_tc_base`.
- **A compute bug fixed** that a ten-minute run exposed: `compute_hours.py` removed only the largest
  idle gap, so `output/composition` (a rolling `--text-out` target) billed 110 idle hours. Total
  261.7 → 376.8 → **265.9**, then 266.2. Disclosure 262 → 266, fine-tunes 52 → 57.

---

## Standing constraints

Never push to a remote. `feat-016` is human-only and must never be started. Do not modify
`~/sub/neurips_2026.tex`, `output.zip`, or the committed prompt sets under `data/`. Never GPU 3;
always `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Ask before fine-tuning above 8B, any run over 24 GPU-hours,
deleting a file the agent did not create, or anything touching a remote. `pgrep -f`/`pkill -f` match
the invoking shell — kill by PID. **Never edit a script while it is running**: bash reads it
incrementally by byte offset, and a commit at 03:17 to a queue started at 02:57 produced a spurious
`FAILED` after all work had completed. A budget violation is per-trajectory. Baselines at `k=-1` and
`k=0` are mandatory. The manuscript is in `~/sub/satml/`, **outside this repo, inside a stray home
git repo that must never be committed to**; after any edit there recompile and check exit status, 0
`??`, 0 overfull, and zero body lines on `pdftotext` page 10. Render a page to PNG and look at it —
it has caught two defects no compile-time check sees.
