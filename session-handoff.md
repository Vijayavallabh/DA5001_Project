# Session handoff — 2026-09-16 07:20 (all four seed ladders SCORED; one sweep still running)

## Current objective

The four-ladder seed arm is **complete and scored**, and it **overturned** a claim this same session
had written into the manuscript six hours earlier. Both corrections are already in the paper. One
memoriser (`memb_pleias_s3`) is still fine-tuning on GPU 2; it cannot change any verdict and is
wanted only for the committed seven-point secondary.

**51 pre-registrations, all scored. 494 tests. `./init.sh` exit 0. Manuscript compiles exit 0, 0
overfull, 0 `??`, 9 of 9 body pages (0 body lines on page 10), 57 total, 3241 numeric literals with
the one expected miss. 261.7 GPU-hours measured.**

---

## READ THIS FIRST: `nvidia-smi` is broken, and the cause was in our own repo

```
nvidia-smi                          -> Failed to initialize NVML: Driver/library version mismatch
env -u LD_LIBRARY_PATH nvidia-smi   -> prints the table
```

`LD_LIBRARY_PATH` leads with `NVIDIA-Linux-x86_64-580.173.02/` — an extracted driver runfile in the
repo root, 1.5 GB, gitignored, **not created by this work** — whose `libnvidia-ml.so.1` shadows the
system's and does not match the loaded kernel module (`580.178.04`).

**CUDA compute is completely unaffected.** `torch.cuda.is_available()` is `True`, every number this
project has measured is correct, and the only casualty is NVML. NVML is what torch's caching
allocator calls inside `generate()`, which is why four fine-tunes died at 05:00 in their
*post-training* check, *after* writing their merged model, and each queue shell then skipped the
sweep behind it (`RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT
FAILED`).

Two repairs, both committed: `e4943f3` made the diagnostic unable to fail a run, and **this session
added `scripts/gpu_env.sh`**, sourced by every GPU launcher, which strips the shadowing directory.
glibc reads `LD_LIBRARY_PATH` once at exec, so a process cannot fix its own search path — the repair
has to be in the shell, before python starts. `tests/test_bookmia_onset.py` pins it by *executing*
it against a poisoned path, not by matching its spelling.

- **Do not delete the directory.** It is not ours (escalation rule: never delete a file the agent
  did not create).
- **`scripts/run_strength_ladder.sh` is NOT yet patched** — it was executing when the fix landed, and
  editing a running bash script corrupts it, because bash reads the file incrementally by byte
  offset. Patch it once its queue exits (one line, copy it from any of the other four).
- Recorded as caution (ab) in AGENTS.md, now **twenty-eight** live cautions.

---

## The result: four seed ladders, and why the fourth changed the paper

One question: does an onset ratio reproduce when nothing changes but `--seed`? Every memoriser on
record in this paper was trained at seed 0, so the paper had never measured it.

| ladder | memoriser, sampled `k=-1` | stop-loss fired? | strength span | **ratio span** |
|---|---|---|---|---|
| KL3M-520M, CopyBench | 0.5149 – 0.5756 | yes, epoch 11/40 every seed | 1.12x | 0.0333 |
| KL3M-520M, BookMIA | 0.6690 – 0.7326 | yes, epoch 26/40 every seed | 1.10x | 0.0663 |
| Pleias-1.2B, CopyBench | 0.9091 – 0.9615 | yes, epochs 29–31/40 | 1.06x | 0.0795 |
| **Pleias-1.2B, BookMIA** | **0.1504 – 0.2045** | **no — 40/40 every seed** | **1.36x** | **0.2597** |

read against feat-121's **0.4721** from varying `--epochs` alone on the same cell.

**After three ladders the reading was "seeds do not move the onset ratio; training length does", and
that sentence was written into `appendix_limitations.tex` and `appendix_robustness.tex` at 06:35.
The fourth ladder refutes it as stated.** Both sentences are now replaced. What all four support:

> The onset ratio is reproducible under a re-seeded recipe **wherever the fine-tune converged and the
> memoriser is strong** (0.033–0.080), and is **not** where the memoriser is marginal (0.2597). In
> neither case is it a property of the pair alone.

The failing ladder fails through *strength*, not through the seed as such: its memoriser barely
clears our own entry gate of `0.10` (caution (a)), it is the only cell where the stop-loss never
fires — all 40 epochs at final loss 0.106–0.114 against a 0.02 threshold and 0.017–0.030 elsewhere —
and it is the only one whose seed moves *strength* by more than 1.12x. Within it the strongest
memoriser carries the lowest ratio, the same sign feat-121's epoch ladder shows.

**The convergence reading is post hoc and is labelled so in both the scoring log and the paper.** The
spans, the entry gate and the stop-epoch diagnostic were all committed before the runs.

**Checked rather than assumed: convergence does NOT explain the nine-pair table's ordering.** Four of
the nine memorisers reached their stop-loss and five did not, and the two groups interleave in rank
(converged 0.8784 / 0.9203 / 0.9933 / 1.0532; not 0.8870 / 0.8916 / 0.9261 / 1.0266 / 1.1658). The
converged four still span 0.1748. Restricting to them does not recover a per-pair reading.

### Band outcomes, each against its own committed pre-registration

| pre-registration | verdict |
|---|---|
| `onset_prediction_seedspread.md` (Pleias BookMIA) | **RUN-TO-RUN VARIATION** — span 0.2597 ≥ the committed 0.20 |
| `onset_prediction_seedspread2.md` (KL3M BookMIA, cross-pair) | **INCONCLUSIVE** — 0.0663/0.2597 = 0.2553, in between, **missing the "below a quarter" band by 0.0014** |
| `onset_prediction_seedspread_copybench.md` | scored 06:35: KL3M fires the narrowing band, Pleias inconclusive |
| `onset_prediction_strength.md` (feat-121, epochs) | scored 02:30: INCONCLUSIVE, `rho = -0.600` |

A band missed by a thousandth is still missed. It was not moved.

### Why the Pleias-BookMIA verdict does not wait on seed 3

The committed quantity is the span over **four** points. A span is a maximum minus a minimum, so
adding a point can only hold or widen it: `0.2597` is a lower bound on the four-point span, which is
therefore already `>= 0.20` whatever seed 3 returns. The verdict is decided; seed 3's number gets
appended when it lands.

---

## What is running, and what to do when it finishes

**GPU 2 only.** `bash scripts/run_strength_ladder.sh seeds 1 2 3`, pid 1616043, started 02:57.
Seeds 1 and 2 are done and scored. Seed 3's memoriser was at epoch 22/40 at 07:20 (~75 s/epoch, it
will run the full 40 because its stop-loss never fires), so: memoriser done ≈ **07:45**, sweep
≈ **11:00–11:45**.

GPU 0 is another user's (63.7 GiB). GPUs 1 and 4 are idle and ours are off them.

**The multi-card window the user opened at 03:13 "for the next 6 hours" expires ≈ 09:13.** After
that the standing rule applies again: **all processes on GPU 2**. It is quoted inside
`run_strength_ladder.sh` and `run_copybench_seeds.sh`, both of which default `GPU` to 2, so the
fallback needs no memory.

### When seed 3's sweep lands — exactly three things, in order

```bash
# 1. append seed 3 to the Pleias BookMIA log (the verdict does not change; the number is recorded)
.venv/bin/python analysis/strength_ladder.py --axis seeds

# 2. the COMMITTED SECONDARY, which needs seven points and has six today
.venv/bin/python analysis/strength_ladder.py --axis pooled
#    at six points it reads rho = -0.543, exact p = 0.2972 (floor 1/2520 at n=7).
#    Do NOT report the six-point value as the secondary: seven is what was registered.

# 3. patch the launcher that could not be patched while it ran
#    add after the `export HF_HUB_OFFLINE=1 ...` line of scripts/run_strength_ladder.sh:
#      . scripts/gpu_env.sh   # (cwd is the repo root) strips the stale in-repo driver from LD_LIBRARY_PATH
```

---

## Zero-GPU work queued, in value order

1. **`sections/appendix_seed.tex` is now the biggest remaining overclaim.** It still presents the
   nine-pair ratios as per-pair properties in three tables (~lines 19–26, ~250, ~311–313), including
   the ">10 words / ≤10 words" split. Four seed ladders and one epoch ladder now say the per-pair
   reading is not supported; `appendix_robustness.tex` and `appendix_limitations.tex` have been
   brought into line and this file has not.
2. **Add a memoriser-strength column to the nine-pair table**, and a convergence marker beside it —
   both are already on disk in each `recipe.json` (`final_loss`, `stop_loss`, `epochs_run`), so it
   costs no compute and lets a reader see the confound directly.
3. **Qualify the seed-word gradient** `rho = -0.958` in `appendix_onset.tex`: it is a correlation
   over nine differently-trained, differently-converged memorisers. `appendix_limitations.tex`
   already carries the caveat; the appendix that states the number does not.

---

## Files changed this session (since `3f047e8`)

| file | why |
|---|---|
| `scripts/gpu_env.sh` (new) | strips the shadowing driver from `LD_LIBRARY_PATH`; the explanation lives here once |
| `scripts/run_bookmia_{memorisers,p1,sweeps}.sh`, `run_copybench_seeds.sh` | source it |
| `tests/test_bookmia_onset.py` | +6 tests; the strip is verified by execution against a poisoned path, and demonstrated to fail on both reintroduced defects |
| `results/onset_prediction_seedspread.md` | SCORED — run-to-run variation |
| `results/onset_prediction_seedspread2.md` | SCORED — inconclusive by 0.0014 |
| `results/strength_ladder_{seeds,kl3m_seeds,pooled}.csv` | the scored ladders |
| `AGENTS.md` | caution (ab), the NVML root cause; count 27 -> 28 |
| `~/sub/satml/sections/appendix_robustness.tex` | three-ladder claim replaced by the four-ladder one; nine-pair convergence audit added |
| `~/sub/satml/sections/appendix_limitations.tex` | same correction, one paragraph |

The manuscript is **outside this repo** and must never be committed to (caution (d)); it has no
version control of its own, so its edits live only on disk.

---

## Standing constraints, unchanged

Never push to a remote. `feat-016` is human-only and must never be started. Do not modify
`~/sub/neurips_2026.tex`, `output.zip`, or the committed prompt sets under `data/`. Never GPU 3
(4 GB T400); always `CUDA_DEVICE_ORDER=PCI_BUS_ID` with `CUDA_VISIBLE_DEVICES`. Ask before
fine-tuning above 8B, any run over 24 GPU-hours, deleting a file the agent did not create, or
anything touching a remote. `pgrep -f`/`pkill -f` match the invoking shell — kill by PID. A budget
violation is per-trajectory, never a mean-bound artefact. Baselines at `k = -1` and `k = 0` are
mandatory. After any edit under `~/sub/satml/`: recompile, check exit status, 0 `??`, 0 overfull,
and count body lines on page 10 of `pdftotext` (must be zero).
