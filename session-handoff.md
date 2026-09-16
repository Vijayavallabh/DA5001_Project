# Session handoff — 2026-09-16 17:55 (read-through done and committed; nothing running; nothing blocking)

## Current objective

**None outstanding.** The convergence crossover is closed in both directions and in the paper, and
the full read-through of the rendered PDF is done and committed at **`3cd2e37`** --- nine defects,
eight fixed, one left as a legible cosmetic nit. No job of ours holds a GPU, and the tree is clean.

**56 pre-registrations, all 56 scored. 538 tests, `./init.sh` exit 0. Manuscript compiles exit 0, 0
overfull, 0 `??`, body inside 9 pages (page 10 carries only uncounted end matter), 55 total. 3,354
numeric literals, one expected `64256` miss. Artifact 901 files. Compute 283.0 GPU-hours, fine-tunes
72.7 over 42 runs.**

> **The manuscript now builds with `newtxtext`, not `times`. Never reinstate `times`.** Under
> tectonic (XeTeX) that package emits **no bold roman at all** --- no warning, exit 0, 0 overfull, 0
> `??` --- so all 90 `\paragraph` headings and 54 `\textbf` emphases rendered as ordinary body text,
> and the body was set in Latin Modern rather than the Times the ICLR template asks for. Controlled
> test: `article` alone and `article`+`iclr2027_conference` both emit `LMRoman10-Bold`; adding
> `times` (or `mathptmx`) emits none; `newtxtext` gives real Times with its bold, italic and
> bold-italic and leaves math in Computer Modern, which is what a pdflatex build of the template
> produces. **The check is one line: `pdffonts iclr_2027.pdf | grep -ci bold` must be `3`, not `1`.**
> Fixing it also took the document from 58 pages to 55.

> **The count of "unscored" logs is a trap.** `grep '^## Scoring'` reports 18 unscored logs; all 18
> are scored under the older `## Scored, <claim>` heading. The authority is
> `tests/test_preregistration_count.py`, whose rule is `"## Scoring, " not in t and "## Scoring log"
> in t` — i.e. a log is unscored only if it still carries the *boilerplate* heading. Do not "fix"
> eighteen files on the strength of a grep.

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

`scripts/gpu_env.sh` strips it and **every GPU launcher sources it**. Do not delete the directory:
it is not ours. Caution (ab) in AGENTS.md.

---

## GPU state, and the window

| card | state |
|---|---|
| 0, 1, 2 | idle, 14 MiB |
| 3 | T400 4 GB — **never use** |
| 4 | **another user**, pid `3703331`, 74 GB, still running at 17:52 (`launch.py --config configs/pfd.yaml`) — leave it alone |

The multi-GPU window opened 10:36 ("for the next 8 hours, use all the gpus") and **expires 18:36
today — about forty minutes from this handoff**. It went **entirely unused after the crossover
closed at 16:05**, and nothing outstanding needs it. It SUSPENDS, not cancels, the one-card rule;
every launcher takes `GPU` as an override **defaulting to 2**, so the standing rule (everything on
GPU 2, in series) restores itself when the window lapses with no action required.
`CUDA_DEVICE_ORDER=PCI_BUS_ID` stays mandatory either way. **Treat the window as closed**: a new
session should assume one card unless the user reopens it.

---

## The convergence crossover: neither direction could be built, and that is the result

The appendices said the onset ratio is reproducible **wherever the memorisation fine-tune
converged**, labelled post hoc because the one irreproducible cell is also the only non-converged
one *and* the only marginal memoriser — three things moving together. Convergence is manipulable, so
the confound was attacked from both sides. Neither side could be built.

| arm | pre-registration | intervention | outcome |
|---|---|---|---|
| forward | `onset_prediction_convergence_causal.md` (40 ep) → `..._causal_60.md` (60 ep) | Pleias-1.2B/BookMIA: lower the rate until the `0.02` stop-loss fires | **INVALID, ABANDONED.** At 60 epochs `lr 1.5e-4` and `1e-4` both plateau at **`0.0222`**, flat for three epochs. A floor the pair cannot cross, with the stop-loss *underneath* it |
| reverse | `..._convergence_reverse.md` → `..._reverse_2.md` (rates 7e-4/8e-4/9e-4) | KL3M-520M/BookMIA: raise it until the stop-loss stops firing | **INVALID, ABANDONED.** `6e-4` converges (`0.0189` at ep 25, `k=-1` 0.855). `7e-4` is chaotic (`0.137` at ep 31, then `4.96`, `4.25` at cap) with measured entry gate **`0.0000`**, `lcs_word` `1.52` — *less than the `1.73` an unrelated anchor manages* |

**What the failure measured, and why it is worth more than the crossover would have been:** *"the
stop-loss never fires"* is not one phenomenon. On Pleias-1.2B the threshold sits **below the pair's
achievable floor**; on KL3M-520M it can be moved only by **destroying the model**, and a 17% rate
change spans the whole distance. That 7e-4 is chaotic rather than dead also means whether a run
yields a memoriser or a wreck depends on where the epoch cap falls.

**No sweep ran and no onset ratio was computed at any point in either arm** — twelve fine-tunes and
three entry-gate probes, ~9.5 GPU-h, all construction. Every abandonment followed a rule written
before the runs, and both retries commit to *abandonment* rather than a third probe.

**In the paper:** `appendix_robustness.tex` carries a paragraph recording both failed interventions
and the two floors; `appendix_limitations.tex` states the convergence association **is not shown to
be causal, and not for want of trying**.

### The test to apply if a stage-1 failure recurs

The 40→60-epoch retry was legitimate and the 60-epoch abandonment was not optional. The line is:
**had one seed been swept, the honest course would have been to stop.** Stage 1 produced four loss
numbers and no measurement, so re-attempting a construction that failed *to construct* is a
different act from re-running an experiment whose answer one dislikes. Both files say so in advance.

---

## The read-through, and the four checks it leaves behind

Everything below was found by **rendering pages to PNG and looking at them**, or by reading a number
back to its CSV. None of it was visible in the source, and the build was clean throughout: tectonic
exit 0, 0 overfull, 0 `??` at every step, including while 144 bold spans were silently not bold.

| check | command | must read |
|---|---|---|
| bold actually renders | `pdffonts iclr_2027.pdf \| grep -ci bold` | **3** (1 means `times` is back) |
| no literal tildes from matplotlib | `pdftotext iclr_2027.pdf - \| grep -c '\.~'` | 0 |
| figures have no collisions | render each figure page to PNG and look | by eye only |
| numbers round from a CSV once | `.venv/bin/python analysis/audit_numbers.py` | 1 miss, `64256` |

**The transferable lesson is about the tests, not the paper.**
`tests/test_reference_targets.py` was written for exactly the defect class that got through, and
caught neither instance. Two reasons, both worth checking in any similar guard: it only looked
**inside parentheses**, so the same defect in running prose was invisible; and its `_resolve()`
walked the section files **independently**, so `sec:onset` --- whose file is `\input` *inside*
`orders.tex` --- resolved to `None`, and every comparison touching it passed vacuously rather than
failing. A guard that returns `None` for the interesting case is a guard that always passes. It now
splices `\input` and walks the document in order, and two added tests were each shown to fail on the
reintroduced defect before being accepted (538 tests, up from 536).

A third instance of the same shape, found while writing those tests: a comment-stripping
`re.sub(r"^\s*%.*$", "", body, flags=re.M)` applied **after** `body.replace("\n", " ")` matches from
the first comment to the end of the string and silently empties the whole document. Strip comments
before joining lines.

---

## Closed this session (14 commits since `3f047e8`)

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
- **The crossover, above.**
- **A full read-through of the rendered PDF** (`3cd2e37`), which is where the bold defect below came
  from. Nine defects: the document-wide font failure, four figure defects, the abstract's conflated
  range, `0.79` sourced from a CI bound rather than an onset, a 3-seed-vs-4-seed comparison, and two
  cross-references the v8 reorder had silently broken. Eight fixed; the ninth is a legible nit.
- **A compute bug fixed** that a ten-minute run exposed: `compute_hours.py` removed only the largest
  idle gap, so `output/composition` (a rolling `--text-out` target) billed 110 idle hours. Total
  261.7 → 376.8 → 265.9 → **283.0** as the crossover's own hours landed. Disclosure 262 → **283**,
  fine-tunes 52 → **73**.

**Five defects caught in my own work, all before they reached a measurement** — worth reading as a
set, because each was found by a different kind of check: a selection helper that iterated
`reversed()` over a smallest-first list and returned the **largest** non-converging rate where the
committed rule says smallest (caught by re-deriving the pick with an explicit sort); an inline
`nohup bash -c` that never started (the tell was **no log file at all**, since a redirect creates one
even on failure — replaced by `scripts/run_entry_gate_probe.sh`); `pgrep -f` matching only its own
invoking shell for the **sixth** time (real evidence came from log mtime and
`nvidia-smi --query-compute-apps`, which also revealed another user had taken GPU 4 and killed the
9e-4 probe); a script edited at 03:17 while a queue started at 02:57 was executing it, producing a
spurious `FAILED` after all work had completed (bash reads by byte offset — now a standing
constraint); and five **estimated** timestamps written into `progress.md`, three of them in the
future, repointed to actual commit times.

---

## What is actually left

Nothing is blocking. In descending value:

1. **Two cosmetic figure nits, deliberately left.** Figure 5's legend has one word ("Common") crossed
   by one dashed curve; there is no clean fix that does not either hide data behind an opaque legend
   or resize the figure and reflow the document. Figure 2's legend has the `K=S(x)` rule through one
   entry. Both are legible. Fix only if the page budget is being reworked anyway.
2. **Two committed CSVs disagree in the fourth significant figure.** `strength_ladder.csv` and
   `onset_ci.csv` hold the same four epochs-ladder onsets and differ (`4.106` vs `4.1078`, `4.0071`
   vs `4.0058`). **The paper follows `onset_ci.csv`, which is correct** --- it is the source that also
   supplies the table's CIs, and it matches on all twelve cells. Nothing tests that the two agree.
   Do not "fix" the paper against `strength_ladder.csv`.
3. **`fineb_kl3m_s1`/`s2` are on disk with sweeps deliberately dropped** as secondary-only by their
   own pre-registration. Reinstating them would need a new pre-registration; it is not a gap.
4. Nothing else. `feat-010`/`011` remain optional and unstarted; `feat-012` is superseded by
   `feat-024`; **`feat-016` is human-only and must never be started.**

### Human-only, and close

**ICLR 2027 abstract registration is Sep 19, 17:30 IST — under three days out; the paper is Sep 25.**
Registration and submission are human steps the agent must never attempt. The manuscript is
submission-ready as it stands.

---

## Standing constraints

Never push to a remote. `feat-016` is human-only and must never be started. Do not modify
`~/sub/neurips_2026.tex`, `output.zip`, or the committed prompt sets under `data/` (the one writable
path there is `data/gutenberg/`). Never GPU 3; always `CUDA_DEVICE_ORDER=PCI_BUS_ID` with
`CUDA_VISIBLE_DEVICES`; check for other users before taking a card. Model caches under repo
`hf_cache/`, never home; keep scratch and logs off `/` and `~`. Ask before fine-tuning above 8B, any
run over 24 GPU-hours, deleting a file the agent did not create, or anything touching a remote.
`pgrep -f`/`pkill -f` match the invoking shell — kill by PID, and kill the reparented CUDA child too.
**Never edit a script while it is running.** A budget violation is per-trajectory, never a mean-bound
artefact. Baselines at `k=-1` and `k=0` are mandatory. Anonymity: the sole sanctioned self-reference
is third-person `\cite{vijayavallabh2026audit}` / "an earlier audit", never "our earlier audit". The
manuscript is in `~/sub/satml/`, **outside this repo, inside a stray home git repo that must never be
committed to** — always run git with an explicit path into `DA5001_Project`. After any edit there
recompile and check exit status, 0 `??`, 0 overfull, and zero body prose lines on `pdftotext` page
10. Render a page to PNG and look at it — it has caught two defects no compile-time check sees.
