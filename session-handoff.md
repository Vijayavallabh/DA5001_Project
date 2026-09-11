# Session handoff — 2026-09-11 (late)

## Current objective

**feat-088 is running and is the only thing in progress.** Everything else is done and verified.

The manuscript was reinvented today (feat-089, `done`). It is no longer an audit with a constructive
appendix; it argues one claim and exhibits a mechanism.

> The obstruction to an inference-time copyright certificate is **metering per token**, not
> budgeting. `K = kT` is denominated in the quantity it protects, so `K/S(x) -> k/s(x)` and never
> improves. A budget spent once on the draw is `log n` and does not grow with the work at all.

## State

| | |
|---|---|
| manuscript | `~/sub/satml/iclr_2027.tex`, *Meter the Draw, Not the Step* |
| build | `exit=0`, `overfull=0`, `unresolved=0`, main text **exactly 9 of 9 pages**, 39 total |
| tests | **260**, all passing |
| numeric audit | 1822 literals, 1 expected miss (`64256`) |
| tree | clean, branch `iclr-2027` |

Sections: 1 intro · 2 theory (Props 1-2, Thm 1, the asymmetry) · 3 onset, 0.35 page · 4 *Three
repairs, and why each fails* · **5 selection anchoring (Prop 3)** · **6 the experiments** · 7 related
· 8-10 limitations, ethics, conclusion. Figure 1 is the thesis in two panels. Every v5 section is
kept beside its replacement as `*_v5_2026-09-11.tex`.

## What is running

```
h1.py --trajectories-per-prompt 64 ... --output-dir output/phase5/sel_anchor64   # PID 1346971
```

64 anchor candidates on the 500 ordinary prompts, on GPU 4. A chained watcher waits on that PID and
then runs `analysis/selection_scaling.py`; its log is `output/logs/sel_scaling.log`. Expect the
generation to finish roughly six hours after 23:35 and the scoring pass to take about an hour more.

**If the chain did not fire,** run it by hand:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_scaling.py --gen-dir output/phase5/sel_anchor64 --out results
```

## Recommended next step

1. **Score feat-088 against its committed bands** in `results/onset_prediction_selection_scaling.md`
   — append to the scoring log, never edit what is above it. Three readings are committed and they
   partition the line, so whatever comes back is reportable:
   - **O1** `gain(64)` vs `gain(8)` on judge B: SCALES / SATURATES / OVEROPTIMISES.
   - **O2** judge C's gain at `n=8`: GENERAL / PARTIAL / JUDGE-SPECIFIC.
   - **O3** the pointwise selector against half the pairwise `+0.081`: DEPLOYABLE / REFERENCE-BOUND.
   The entry gate is that the `n=1` control lands within `0.05` of each judge's `u_safe` on record
   (`0.440` for judge B). Outside it the run is a pipeline failure and is reported as one.
2. **Integrate whichever way it lands.** Section 6 currently has one value of `n` for the utility
   arm, which is the paper's most obvious remaining gap. A curve in `n` belongs in
   Table~\ref{tab:selection} and in Figure 1(b), which already plots the selection arms and will
   simply gain points. If O1 reads OVEROPTIMISES that is a *better* result than SCALES for the
   paper's argument, because it bounds the mechanism honestly; say so rather than burying it.
3. **Then:** update the compute figure in the LLM Usage statement (currently `136` GPU-hours;
   `analysis/compute_hours.py` reads `output/logs/*.log` and will pick feat-088 up), rerun
   `analysis/audit_numbers.py`, and rebuild the artifact with `scripts/build_artifact.sh artifact`.

## Two cautions this session added

- **Killing a `h1.py` parent does not stop the work.** The CUDA child is reparented to init and keeps
  running on the GPU. An orphan of a cancelled generation competed with its own replacement for an
  hour today. After killing a generation read
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and kill the child by PID.
- **`textwrap.fill` breaks words at hyphens**, and `per-\ntoken` in a LaTeX source renders as
  `per- token`. Programmatic rewraps of manuscript prose must pass `break_on_hyphens=False`; check
  with `grep -n '[a-zA-Z]-$' sections/*.tex` afterwards.
