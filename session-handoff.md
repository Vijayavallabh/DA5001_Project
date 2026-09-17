# Session handoff — 2026-09-17 14:30 (a second reviewer; two arms IN FLIGHT on GPU 4 against the two claims that most need it)

## IN FLIGHT — two arms running, bands committed, neither scored

A second, fuller review arrived (Soundness 3/4, **Presentation 1/4**, Contribution 2/4, **4/10
Reject**). Working its points by depth rather than breadth. Two arms are on GPU 4 now, and **both
can cost the paper a claim that is currently in the abstract**:

- **`results/onset_prediction_sparse_causal.md` (feat-125)** — the sparse causal policy Proposition 3
  *permits* and the paper never built, which the reviewer names as the reason "cannot be repaired"
  is not a theorem. `--initial-bank` (feat-092, built and never run) gives a sequence budget constant
  in T; new `--spend-threshold` decides *where* it goes, causally. Grid: B ∈ {log 8, log 64, 64} at
  τ=0, and τ ∈ {1,2,4,8} at B=log 64. If a causal policy matches selection at matched budget,
  **"per-token metering cannot be repaired" is withdrawn.**
- **`results/onset_prediction_alpha_trivial.md` (feat-126)** — is α=8's 80× extraction cut a repair
  that works, or a walk onto the trivial horn? The reviewer reads Table 3 as evidence FOR Repair 1
  and is arithmetically right; the paper's threshold answer does not respond to them. Judged utility
  of α=8 at k=3 against the anchor alone, 500 prompts. If α=8 holds its utility, **"the three repairs
  fail where it says they must" is false as written.**

- **`results/onset_prediction_paraphrase.md` (feat-127)** — the paraphrase claim, which the reviewer
  correctly says no experiment supports. Every extraction number in the paper is an *exact substring*
  metric; `rouge_l_score` (LCS as a **subsequence**) has existed in `dap/stats.py` all along and the
  extraction arm never imported it. Measures the anchor's own base rate at a non-literal event —
  which is exactly the quantity the certificate multiplies. **Queued behind the two above on GPU 4.**

## Current objective

**None outstanding.** A reviewer (Reject 5 → Weak Accept 6) named three fixes to close the deal and
all three are done as *experiments*, not as prose: **R1** the contamination screen at OLMo-2 scale
(`feat-122`), **R2** an independent repeat of the head-to-head (`feat-123`), **R3** per-arm
GPU-seconds (`feat-124`). Each had its bands committed before it ran; **two of the three returned
something against the paper and the text was changed accordingly.** Tree clean at **`305d82a`**,
nothing of ours on a GPU.

**59 pre-registrations, all 59 scored. 549 tests, `./init.sh` exit 0. Manuscript compiles exit 0,
with 0 overfull boxes and 0 underfull at badness 10000 (158 hboxes → 13), 0 `??`,
`pdffonts | grep -ci bold` = 3, body inside 9 pages (the Ethics Statement opens on page 9), 57
total. 3,489 numeric literals, one expected `64256` miss. Artifact 939 files. Compute 283.0 → 295.3
GPU-hours as the three arms landed; the disclosure moved with it.**

> **A second full read-through is done (2026-09-17 15:40), and it found eleven defects, none of them
> visible in the source.** The build was clean at every step. The serious one: `experiments.tex`
> still carried the **withdrawn** cross-protocol vetting claim ("all eighteen models ... every model
> known to hold the work at least half") that the appendix and the Ethics Statement had already
> dropped, because the test written for that repair only scanned the appendix. Also four figure
> collisions (Figures 1, 3, 6), **two figures printed at half the size they were drawn** (Figures 5
> and 6, shrink `0.494` and `0.638`), three quotation-mark defects in one class, two cross-references
> pointing at sections that do not contain the number, one sentence fragment, and a FLOP cost table
> the latency measurement had left unqualified. All fixed; `progress.md` has the full account and
> AGENTS.md caution **(af)** the reusable lessons. **Its one deferred item is closed too**: the 158
> underfull hboxes (58 visibly stretched) came down to 13, none at badness 10000, by letting the 38
> long `\texttt{}` paths break after `/` and `\_` — see "What is actually left", item 1, for the two
> traps that pass hit.

### What the three arms changed in the paper

| | asked | answered | what moved in the text |
|---|---|---|---|
| **R1** | screen OLMo-2/DCLM | **Band A holds, Band B fails.** Five licensed anchors `0.000`, OLMo-2-7B `0.040`, OLMo-2-13B `0.120`, 70B control `0.500` | the licensing premise is **measured**, not asserted; Section 3, the vetting table and the Ethics Statement rewritten; DCLM substituted (every checkpoint declares `model_type: openlm`), recorded before any model ran |
| **R2** | repeat the head-to-head, ideally judge C | **all three new D3 estimates CONFIRMED**; the fresh draw reproduces `+0.0645` to `0.001` | new Appendix I paragraph `app:h2hrepeat` with the four-row table; a registered expectation (judge C would be harsh) **withdrawn**; arm B's one-sided independence stated as a limit |
| **R3** | per-arm GPU-seconds | **MODEL HOLDS at `R=35.4x`** (band 30–123) **but the secondary FIRES**: the reward pass is `9.3%` of selection's clock, the draws `90.7%` | the paper's "`61.3x` is the price of the reward model, not the mechanism" was **wrong on a clock** — corrected in Section 5, Appendix I (`app:latency`) and the closing; the abstract now quotes the measured ratio |

> **R3 is the one to read first if you are picking this up.** The FLOP model is not merely
> imprecise, it is **backwards about where the money goes**: scoring 64 candidates is one batched
> forward pass, drawing them is `64 x 204` sequential decode steps. *With a free scorer selection
> would still cost `32.1x`.* So "a smaller scorer cuts the price" is true of the FLOP count and close
> to false of the clock — a `1.5B` scorer can return at most the `9.3%` it occupies. **The lever on
> serving cost is `n`.** Where the FLOP claim survives it now names FLOPs as its currency, and
> `tests/test_h2h_repeat_and_latency.py` refuses a version that does not. Caution **(ae)**.

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

Read at 17:20 with `env -u LD_LIBRARY_PATH nvidia-smi` (caution (ab) — the bare command returns
nothing). **Two cards have just come free**, which was not true at 15:50:

| card | state |
|---|---|
| 1, 2 | **idle**, 14 MiB — available, but re-read before taking one |
| 0 | **another user**, 597 MiB idle — not ours |
| 4 | **another user**, 65 GB at 60% util — leave alone |
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

## Two read-throughs, and what they leave behind

Twenty defects across two passes (nine on 2026-09-16, eleven on 2026-09-17), **every one of them
found by rendering pages to PNG and looking at them**, or by reading a number back to its CSV. The
build was clean throughout both: tectonic exit 0, 0 overfull, 0 `??` at every step --- including
while 144 bold spans were silently not bold, and while Section 3 asserted a claim the appendix had
withdrawn two sections later.

### The standing check list

| check | command | must read |
|---|---|---|
| bold actually renders | `pdffonts iclr_2027.pdf \| grep -ci bold` | **3** (1 means `times` is back) |
| no literal tildes from matplotlib | `pdftotext iclr_2027.pdf - \| grep -c '\.~'` | 0 |
| figures have no collisions | render each figure page to PNG and look | by eye only |
| legend style keys show their dashes | look at the key, not just the labels | `handlelength` >= 3 |
| **figure type is big enough** | `printed width / figsize width` per figure | **>= 0.7** (see (af)) |
| numbers round from a CSV once | `.venv/bin/python analysis/audit_numbers.py` | 1 miss, `64256` |
| quotation marks curl the right way | `pytest tests/test_contaminated_anchor.py` | no ASCII `"` and no markdown `` ` `` |
| long `\texttt` paths can break | same file, `test_long_texttt_paths_carry_breakpoints` | `\allowbreak` after **every** `/` and `\_` |
| no visibly stretched lines | `grep -c 'Underfull .hbox (badness 10000)' <log>` | **0** |
| a withdrawn claim stays withdrawn | `pytest tests/test_anchor_vetting.py` | scans **every** section |

### Pass 2 (2026-09-17): eleven defects, and the body contradicted the appendix

**The serious one.** `experiments.tex` still read *"the check separates all eighteen models here ...
and every model known to hold the work at least half"* --- the **withdrawn** cross-protocol claim.
`feat-122` had rebuilt that table into two protocol blocks, and at the one registered protocol
OLMo-2-7B's worst passage is `0.2677`, nowhere near half. The appendix and the Ethics Statement were
corrected when the arm landed; **Section 3 was not, because the test written for that repair only
scanned `appendix_selection.tex`.** A test written for a repair must scan every section file, not
the one the arm was about. Now it does, and it also asserts the *positive* half still holds of the
CSV (every licensed anchor exactly zero, every web-trained model above it).

**Figures: six problems in four figures.** Four collisions of the caution-(ad) kind --- Figure 1(b)'s
unframed legend sat under the $\Lambda^*$ curve and struck out "Thm. 1" and "selection anchoring";
its `n=8` was struck by the blue curve and `k=0.5` by the axis spine; Figure 1(a)'s `k=0.5` by the
median-target rule; Figure 3(a)'s legend by the axvline; Figure 6's "certificate vacuous" by a curve.
**Moving a legend trades one collision for another:** relocating Figure 1(b)'s to the empty upper
right put its frame through the `n=64` label, and the panel needed `ylim` headroom as well.

**And two figures were printed at half the size they were drawn.** `frontier_scaling` (Fig 5) was
`width=0.62\textwidth` on a `figsize=(6.9, ...)` canvas --- a shrink of `0.494`, so its 7pt legend
printed at `3.5`pt --- and `onset_collapse` (Fig 6) at `0.8` was `0.638`. Both now `\textwidth`.
`selection_frontier`'s own source comment said "if this figure must get narrower, shrink figsize too
and RENDER THE PAGE" and it was not followed for its neighbours. It is the only 6.9in figure that
may be narrower, because it pre-scales type by `F = 6.9/5.5`.

**Three quotation-mark defects, one class.** ASCII `"..."` renders as two *closing* quotes (mine, in
the new latency paragraph); markdown `` `factual` ``/`` `creative` ``/`` `neutral` `` and
`` `analysis/seed_effect.py` `` as two *opening* ones; and `references.bib`'s `Probabilistic
"Copies"` put two closing quotes in the reference list. Now one test greps both shapes --- and its
first regex was too permissive (it matched across a legitimate `` ``Active'' ... ``loss'' `` pair)
and its second missed a path, both caught by *running* it rather than reading it.

**Two cross-references pointed at sections that do not contain the number:** `849` is stated in
Section 2, not Section 5; the intro's new cost clause sent a reader to Section 3 for `61.3x`/`35.4x`,
which live in Section 2 and Appendix I. One sentence fragment of mine --- *"Measured on one card that
is $35.4\times$ the wall-clock"* --- parsed as a relative clause on "card".

**A residual the latency arm left behind**, found by reading the appendix rather than by grep: the
scorer-free table prices majority vote at `5.75x` against the reward model's `61.29x`, and that
column is **FLOPs**. On the clock the scorer is `9.3%`, so dropping it saves far less than dropping
parameters suggests. The column is now labelled `cost, FLOPs`, and a new sentence says we measured
the clock at **one point only** (`n=64`, the `7.6`B scorer) and **do not extrapolate it down the
column**.

### Pass 1 (2026-09-16): nine defects, including a document-wide font failure

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

## Closed this session (40 commits since `3f047e8`)

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
- **The underfull lines, fixed** (`158 -> 13`, none at badness 10000). One cause for all of them:
  a 50-character `\texttt{}` path is one unbreakable token. `\allowbreak` after every `/` and `\_`
  in 38 long paths; it prints nothing, so no break can read as a hyphen. Extending the rule to
  hyphens made it worse and was reverted; the one hyphen break needed is by hand.
- **A second full read-through** (`dae7cf8`), after the three arms had changed the abstract and added
  two appendix tables. **Eleven defects, all fixed** — the withdrawn vetting claim still standing in
  Section 3, six figure problems in four figures (four collisions, two figures printed at half the
  size they were drawn), three quotation-mark defects, two cross-references pointing at sections that
  do not contain the number, one sentence fragment, and a FLOP cost table the latency measurement had
  left unqualified. Details above; the two reusable lessons are caution **(af)**.
- **The reviewer's three top fixes, all as experiments** (`feat-122`/`123`/`124`, sections above).
  Twelve commits, ~13 GPU-hours, three pre-registrations each scored against bands committed before
  its run. Two of the three landed *against* the paper and the text moved: the licensing premise
  stopped being an assertion, and the cost claim was corrected in four places including the abstract.
  Also **repaired a defect in the paper's newest contribution** on the way in — `anchor_vetting.csv`
  had compared two protocols and called it a separation between models.
- **Compute disclosure `283` → `295` GPU-hours** (`iclr_2027.tex:163`), because the three arms ran.
  `analysis/compute_hours.py` rewrites `results/compute_hours{,_summary}.csv` every time it is run
  and `tests/test_compute_hours.py` reads the manuscript back, so the two cannot drift silently.
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

## SCORED: the independent head-to-head (`onset_prediction_h2h_independent.md`, `feat-123`)

Arm A finished 22:53 on GPU 2, arm B 02:11 on GPU 1. The registration fixed the verdict rule first:
the headline stands as written **only if all three** new estimates are positive with intervals
excluding zero.

| draw | judge | D1 selection | D2 metered | **D3** | verdict |
|---|---|---|---|---|---|
| 42 (original) | B | `+0.1045` | `+0.0400` | **`+0.0645`** `[+0.0300,+0.0995]` | — |
| 42 | C | `+0.1360` | `+0.0740` | `+0.0620` `[+0.0165,+0.1075]` | **CONFIRMED** |
| 52 (fresh) | B | `+0.1035` | `+0.0400` | `+0.0635` `[+0.0290,+0.0975]` | **CONFIRMED** |
| 52 (fresh) | C | `+0.1740` | `+0.0740` | `+0.1000` `[+0.0520,+0.1465]` | **CONFIRMED** |

All three confirm, and **the fresh draw at the same judge reproduces the original to `0.001`** —
a stronger answer than the arm was designed to give, since `build_trajectory_seeds` hashes the
`--seeds` tuple into the high 16 bits, so `52,53,54` gives 64 trajectory seeds disjoint from
`42,43,44` by construction. **Three things reported against us rather than folded in:** the
seed-52/judge-C estimate falls just outside the original interval on the *favourable* side and is
quoted separately; the registration's claim that judge C would be a **harsh** test (it is the fixed
opponent's own checkpoint, caution (aa)) was **wrong** — it raised both gains — and is withdrawn in
the paper; and arm B regenerates the **selection** arm only, which is why D2 is identical down each
judge column, so the independence is generation-level on one side and judge-level on both.

A third "different bootstrap seed" axis was considered and **rejected as fake**:
`order_averaged_h2h.py --seed` drives only `paired_boot`, not the judging and not the presentation
order, so it would jiggle CI endpoints and prove nothing.

`order_averaged_h2h.py` writes **fixed filenames** and would have clobbered the committed original.
It was backed up to `*_seed42_judgeB.csv` before any run, each launcher restores the canonical file
at the end, and `diff` is clean — `git diff results/order_averaged_h2h.csv` is empty.

---

## SCORED: what `61.3x` costs on a clock (`onset_prediction_serving_latency.md`, `feat-124`)

Ran 22:53--23:25 on GPU 2, exclusively ours, interleaved `SEL,MET,MET,SEL` so drift cancels. Both
arms serve one completion per prompt over the same 40 prompts — **8,179 served tokens each**, so the
ratio of seconds-per-served-token *is* the wall-clock ratio, and `serving_latency.py` asserts the
denominators are equal rather than assuming it.

| path | repeats (s) | mean (s) | s / served token |
|---|---|---|---|
| selection n=64, the 64 anchor draws | 824.6, 822.6 | 823.60 | 0.1007 |
| selection n=64, the reward pass | 85.8, 83.1 | 84.44 | 0.0103 |
| selection n=64, total | 910.4, 905.7 | 908.03 | 0.1110 |
| metered k=10 | 23.7, 27.6 | 25.64 | 0.0031 |

**Primary `R = 35.42x`, inside the committed 30–123 band → MODEL HOLDS**, but the FLOP model is
`1.73x` pessimistic as a price. **Secondary FIRES**, and it is the finding that matters: the
committed rule was *"if the reward pass is less than half the measured selection time, the 'price of
the reward model, not of the mechanism' argument is weakened and the text must say so"* — it is
`9.3%`. Repeat spread `0.52%` (SEL) and `15.27%` (MET, i.e. 3.9 s on a 25 s job, which is why both
repeats are printed); extremes leave `R` in `[32.8, 38.5]`.

Two design rules the measurement depended on, both now in caution (ae): equal denominators by
construction, and the served-token count from `aggregate.generation_length_tokens` (the **decoded**
length) rather than the per-step log, which is padded past the end of the generation (caution (s)).

---

## What is actually left

Nothing is blocking. In descending value:

1. **Nothing typographic.** The `Underfull \hbox` item that stood here is closed: all 158 had one
   cause --- a 50-character `\texttt{results/onset\_prediction\_...}` is a single unbreakable token,
   so TeX moved it whole to the next line and stranded the one before it. `\allowbreak` after every
   `/` and `\_` in long `\texttt` arguments (38 paths, 6 files) took it to **13, none at badness
   10000**, with 0 overfull throughout. It prints nothing, so a reader cannot mistake the break for
   a hyphen. The 13 that remain are two paragraphs TeX genuinely cannot set better: a proof carrying
   long inline `$D_{\mathrm{KL}}$` expressions, and one bibliography entry. **Do not extend the rule
   to hyphens** --- it was tried and took badness-10000 lines from 0 back to 7, because more
   breakpoints move TeX's choices elsewhere; the one hyphen break needed is done by hand in
   `appendix_proofs.tex`. **And any consumer that greps paths out of the manuscript must strip
   `\allowbreak` first**, which is how `test_abstract_consistency.py` started reading a filename as
   `\allowbreak`. Pinned by `test_long_texttt_paths_carry_breakpoints`, which checks every
   separator.
2. **Two committed CSVs disagree in the fourth significant figure.** `strength_ladder.csv` and
   `onset_ci.csv` hold the same four epochs-ladder onsets and differ (`4.106` vs `4.1078`, `4.0071`
   vs `4.0058`). **The paper follows `onset_ci.csv`, which is correct** --- it is the source that also
   supplies the table's CIs, and it matches on all twelve cells. Nothing tests that the two agree.
   Do not "fix" the paper against `strength_ladder.csv`.
3. **The reviewer's remaining points, deliberately not taken.** The instruction was depth over
   breadth, so three were worked to completion and the rest were judged not worth the space:
   a third judge (B and C are the only two available --- judge A supplies the selection reward and
   may not score the arm it selected), a second protected corpus for the selection sections (the
   nine-pair onset work already runs on two), and a formal treatment of the approximation gap
   between the optimal budget-`K` policy and our causal decoder, which the paper states as its open
   problem rather than closing. Each is a real gap; none is a defect.
4. **`fineb_kl3m_s1`/`s2` are on disk with sweeps deliberately dropped** as secondary-only by their
   own pre-registration. Reinstating them would need a new pre-registration; it is not a gap.
5. Nothing else. `feat-010`/`011` remain optional and unstarted; `feat-012` is superseded by
   `feat-024`; **`feat-016` is human-only and must never be started.**

### Files changed since `9a76354`

**Repo** (committed, tree clean at `305d82a`):
`analysis/{anchor_vetting,serving_latency}.py` · `scripts/run_{vetting_protocol,vetting_split,h2h_independent,h2h_judgec,serving_latency}.sh` ·
`tests/{test_anchor_vetting,test_h2h_repeat_and_latency}.py` ·
`results/onset_prediction_{vetting_protocol,h2h_independent,serving_latency}.md` (registered, then scored) ·
`results/{anchor_vetting,serving_latency}.csv`, `results/vet_*`, `results/order_averaged_h2h_seed{42_judgeB,42_judgeC,52_judgeB,52_judgeC}.csv`,
`results/selection_{rewards64,scaling}_seed52.csv` · `results/compute_hours{,_summary}.csv` ·
`AGENTS.md` (caution (ae), count → thirty-one) · `feature_list.json` (feat-122/123/124) ·
`progress.md` · `session-handoff.md` · `artifact/` (939 files).

**Manuscript** (`~/sub/satml/`, never committed — it sits inside a stray home git repo):
`iclr_2027.tex` (abstract cost clause, compute disclosure, pre-registration count) ·
`sections/appendix_selection.tex` (vetting table rebuilt as two protocol blocks, `app:h2hrepeat`
and `app:latency` added, the FLOP claims qualified and the cost column labelled) ·
`sections/selection.tex` · `sections/iclr_intro.tex` · `sections/iclr_closing.tex` ·
`sections/experiments.tex` (the withdrawn vetting claim) · `sections/frontier.tex` (the `849`
reference) · `sections/appendix_robustness.tex` (Figures 5 and 6 widened to `\textwidth`, markdown
backticks) · `sections/appendix_seed.tex` (a backticked path) · `references.bib` (one title's
quotes).

**Figures** (repo, regenerated and copied): `figures/make_figures_v4.py` — Figure 1 (legend moved and
made opaque, three label offsets, `ylim` headroom, the `k=0.5` anchor), Figure 3 (opaque legend),
Figure 6 (opaque annotation bbox).

**Line-breaking pass** (manuscript, 2026-09-17 17:10): `\allowbreak` after every `/` and `\_` in the
38 long `\texttt` paths across `iclr_2027.tex`, `appendix_proofs`, `appendix_robustness`,
`appendix_seed`, `appendix_selection` and `appendix_limitations`, plus two hyphenated model names in
`appendix_proofs.tex` by hand.

**Tests added or repaired this session** (repo): `tests/test_h2h_repeat_and_latency.py` (new, 5) ·
`tests/test_anchor_vetting.py` (+1, the withdrawn claim across every section) ·
`tests/test_contaminated_anchor.py` (+2, quotation-mark direction and path breakpoints) ·
`tests/test_abstract_consistency.py` (parser taught to strip `\allowbreak`). 546 → **549**.

### Recommended next step

**Nothing.** Every thread this session opened is closed: the reviewer's three top fixes are run and
scored, the read-through is done, and its one deferred item is now fixed too. The manuscript is
submission-ready --- 0 overfull, 0 underfull at badness 10000, 0 `??`, bold renders, body inside 9
pages, every number rounding once from a committed CSV, and 549 tests over it.

If someone picks this up with time to spend, the honest answer is that the remaining work is
**scientific, not editorial**, and it is named in item 3 below: a third judge is unavailable by the
paper's own rule, a second protected corpus for the selection sections would cost a fresh
pre-registration and a GPU day, and the approximation gap between the optimal budget-`K` policy and
our causal decoder is stated as the open problem rather than closed. None is a defect.

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
