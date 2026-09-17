# Session handoff — 2026-09-17 12:20 (the reviewer's three top fixes are all run and scored; nothing of ours is running; all four A100s belong to other users)

## Current objective

**None outstanding.** A reviewer (Reject 5 → Weak Accept 6) named three fixes to close the deal, and
all three are done as experiments rather than as prose: **R1** the contamination screen at
OLMo-2 scale (`feat-122`), **R2** an independent repeat of the head-to-head (`feat-123`), **R3**
per-arm GPU-seconds (`feat-124`). Each had its bands committed before it ran; two of the three
returned something **against** the paper and the text was changed accordingly. Tree clean, nothing
of ours on a GPU.

**59 pre-registrations, all 59 scored. 546 tests, `./init.sh` exit 0. Manuscript compiles exit 0, 0
overfull, 0 `??`, `pdffonts | grep -ci bold` = 3, body inside 9 pages (the Ethics Statement opens on
page 9), 56 total. 3,483 numeric literals, one expected `64256` miss.**

### What the three arms changed in the paper

| | asked | answered | what moved in the text |
|---|---|---|---|
| **R1** | screen OLMo-2/DCLM | **Band A holds, Band B fails.** Five licensed anchors `0.000`, OLMo-2-7B `0.040`, OLMo-2-13B `0.120`, 70B control `0.500` | the licensing premise is **measured**, not asserted; Ethics Statement rewritten; DCLM substituted (every checkpoint declares `model_type: openlm`), recorded before any model ran |
| **R2** | repeat the head-to-head, ideally judge C | **all three new D3 estimates CONFIRMED**; the fresh draw reproduces `+0.0645` to `0.001` | new Appendix I paragraph with the four-row table; a registered expectation (judge C would be harsh) **withdrawn**; arm B's one-sided independence stated as a limit |
| **R3** | per-arm GPU-seconds | **MODEL HOLDS at `R=35.4x`** (band 30–123) **but the secondary FIRES**: the reward pass is `9.3%` of selection's clock, the draws `90.7%` | the paper's "`61.3x` is the price of the reward model, not the mechanism" was **wrong on a clock** and is corrected in Section 5, Appendix I and the closing; the abstract now quotes the measured ratio |

> **R3 is the one to read first if you are picking this up.** The FLOP model is not merely
> imprecise, it is **backwards about where the money goes**: scoring 64 candidates is one batched
> forward pass, drawing them is `64 x 204` sequential decode steps. *With a free scorer selection
> would still cost `32.1x`.* So "a smaller scorer cuts the price" is true of the FLOP count and close
> to false of the clock — a `1.5B` scorer can return at most the `9.3%` it occupies. **The lever on
> serving cost is `n`.** Anywhere the FLOP claim survives it now names FLOPs as its currency, and
> `tests/test_h2h_repeat_and_latency.py` refuses a version that does not.

> **Both new tables are generated from their CSVs by a test, not transcribed.** They had to be: the
> first draft of the repeat table rounded four intervals a *second* time and put two of them one out
> in the last digit (caution (j), again), and both scoring logs quoted the extreme-ratio pair as
> `[32.9, 38.3]` where the timings give `[32.8, 38.5]`. Every one of the five assertions was checked
> to **fail** under a deliberate one-digit mutation before being kept — the FLOP-currency guard
> passed vacuously off a "forward-pass FLOPs" two sentences downstream until its window was
> tightened to ±70 characters.

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

## GPU state

Read at 12:18 with `env -u LD_LIBRARY_PATH nvidia-smi` (caution (ab) — the bare command returns
nothing):

| card | state |
|---|---|
| 0, 1, 2, 4 | **all four in use by other users**, 51–57 GB each at 53–64% util — take none of them without checking again |
| 3 | T400 4 GB — **never use** |

**Nothing of ours is running.** The multi-GPU instruction of 2026-09-17 ("use all the 3 gpus to
their fullest vram") applied to the R2/R3 arms, which finished at 02:11 and 23:25. With the box now
fully occupied, the next GPU arm waits; `CUDA_DEVICE_ORDER=PCI_BUS_ID` stays mandatory and GPU 3 is
never used.

**Caution (c) took two more incidents this session**, both mine, and the second was caught only by a
guard. A waiter written as `until ... ! pgrep -f 'snapshot_download'` can never exit, because the
string is in the polling shell's own command line — it spun 93 minutes past a finished download. And
`pgrep -u $USER -f 'order_averaged_h2h.py'` returned my *invoking shell*; had that PID been passed to
the latency launcher, R3 would have started immediately, shared a card, and voided the very
measurement it exists to make. The pattern that works is

```
ps -eo pid,args --no-headers | awk '$2 ~ /python$/ && /<script>\.py/ {print $1; exit}'
```

followed by asserting the PID's argv before waiting on it with `kill -0`.

---|---|
| 0 | **another user**, 569 MiB — not ours, check before taking it |
| 1, 2 | idle, 14 MiB |
| 3 | T400 4 GB — **never use** |
| 4 | **another user**, 59.6 GB, 64% util (`launch.py --config configs/pfd.yaml`) — leave it alone |

**The multi-GPU window has lapsed.** It opened 10:36 ("for the next 8 hours, use all the gpus") and
**expired 18:36 today**, having gone entirely unused after the crossover closed at 16:05. It
SUSPENDED rather than cancelled the one-card rule, and every launcher takes `GPU` as an override
**defaulting to 2**, so the standing rule restored itself with no action: **everything on GPU 2, in
series**, unless the user reopens the window. `CUDA_DEVICE_ORDER=PCI_BUS_ID` stays mandatory either
way, and GPU 3 is never used.

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

## The read-through, and what it leaves behind

Everything below was found by **rendering pages to PNG and looking at them**, or by reading a number
back to its CSV. None of it was visible in the source, and the build was clean throughout: tectonic
exit 0, 0 overfull, 0 `??` at every step, including while 144 bold spans were silently not bold.

| check | command | must read |
|---|---|---|
| bold actually renders | `pdffonts iclr_2027.pdf \| grep -ci bold` | **3** (1 means `times` is back) |
| no literal tildes from matplotlib | `pdftotext iclr_2027.pdf - \| grep -c '\.~'` | 0 |
| figures have no collisions | render each figure page to PNG and look | by eye only |
| legend style keys show their dashes | look at the key, not just the labels | `handlelength` >= 3 |
| numbers round from a CSV once | `.venv/bin/python analysis/audit_numbers.py` | 1 miss, `64256` |

**Five figure collisions, and no two wanted the same repair.** All are fixed; the value left behind
is the diagnosis, because "move the legend out" is wrong for three of the five.

| figure | what was wrong | repair |
|---|---|---|
| 7 | 12-entry legend taller than its own 2.2in panel; covered the panel, its title and the y-label | legend below the panels |
| 6 | 9 entries with 9 dotted `s(x)` rules through them; annotation printed under two of them | legend below the panels |
| 9 | a label that has to stay beside the `y=1` rule it names | opaque `bbox` |
| 5 | labels spelled out what colour and linestyle already encoded | **factor** 6 into 3 + 2, then share |
| 2 | legend pinned to `loc="lower right"` --- the corner the `K=S(x)` rule passes through | a different corner |

Figure 5 is the one to remember: moving it out would have resized the figure and reflowed the
document, and an opaque legend would have hidden most of a dashed curve. Shortening the labels cost
nothing and removed a duplicate legend as well. Also check the *keys*, not just the label text --- at
`handlelength=1.8` a style key's marker covers its handle and solid reads the same as dashed.

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

## Closed this session (24 commits since `3f047e8`)

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
- **A full read-through of the rendered PDF** (`3cd2e37`, then `848d201` and `0b93109`), which is
  where the bold defect above came from. **Nine defects, all fixed**: the document-wide font
  failure, five figure defects, the abstract's conflated range, `0.79` sourced from a CI bound
  rather than an onset, a 3-seed-vs-4-seed comparison that broke a rule the paper states two pages
  earlier, and two cross-references the v8 reorder had silently broken. The last two figures
  (5 and 2) were fixed after the main commit, on request.
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

## SCORED: the vetting-protocol arm (`onset_prediction_vetting_protocol.md`)

Ran 19:11--22:23 on GPUs 1 and 2, ~9.8 GPU-hours, **scored 22:30**. Nothing of ours is running now.

**It repaired a defect in the paper's newest contribution and answered the reviewer's scale
question in one arm.** `anchor_vetting.csv` had compared five anchors screened at ~14 genuine prefix
tokens against a control screened at 100 raw ones; the same `Llama-3.1-70B` reads `0.0000` at the
anchors' protocol, so "separation complete over eighteen models" was a separation between protocols.
One protocol, every model in the anchor slot, seeds verified byte-identical to the control's
(`sha256 8b46267d75766de1` both sides):

| model | leaking | max recall | band |
|---|---|---|---|
| the five licensed anchors | **0.000** | 0.0000 | **A: INSTRUMENT HOLDS** |
| OLMo-2-7B | 0.040 | 0.2677 | **B: FAILS** |
| OLMo-2-13B | 0.120 | 0.5659 | **B: FAILS** |
| Llama-3.1-70B (control) | 0.500 | 1.0000 | — |

Band B failing is the outcome that *helps*: the licensing-frontier argument in Section 3 is now a
measurement rather than an assertion, which is what the band committed to in advance. Manuscript
updated in three places (Section 3, `appendix_selection.tex`'s table and prose, the Ethics
Statement). DCLM is absent — every published checkpoint is `openlm` format, unloadable without a new
dependency; OLMo-2-13B replaced it, recorded **before any model ran**.

**Not registered, do not restate as a result:** contamination rises with scale across the three
web-trained models (0.040 / 0.120 / 0.500 at 7B / 13B / 70B). Three points, confounded with corpus
and recipe.

---

## What is actually left

Nothing is blocking. In descending value:

1. **Two committed CSVs disagree in the fourth significant figure.** `strength_ladder.csv` and
   `onset_ci.csv` hold the same four epochs-ladder onsets and differ (`4.106` vs `4.1078`, `4.0071`
   vs `4.0058`). **The paper follows `onset_ci.csv`, which is correct** --- it is the source that also
   supplies the table's CIs, and it matches on all twelve cells. Nothing tests that the two agree.
   Do not "fix" the paper against `strength_ladder.csv`.
2. **`fineb_kl3m_s1`/`s2` are on disk with sweeps deliberately dropped** as secondary-only by their
   own pre-registration. Reinstating them would need a new pre-registration; it is not a gap.
3. Nothing else. `feat-010`/`011` remain optional and unstarted; `feat-012` is superseded by
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
