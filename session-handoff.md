# Session handoff --- 2026-09-19 evening

## Current objective

The appendix reduction is **finished**: 46 -> 32 pages, appendix 33 -> **20** (pp. 13--32), main text
untouched at exactly 9 of 9 with the Ethics Statement at the top of page 10. 667 tests pass, exit 0,
0 overfull, 0 `??`. Nothing about the manuscript is outstanding except folding in the two GPU arms
below when they land.

## What was done

1. **Appendix 33 -> 20 pages, in four tranches** (commits `7367b16`, `5b96f53`, `f7bab87`,
   `7e062f1`). The rule throughout: **delete the apparatus, keep every concession, every guarded
   number and every claim the main text points at.** Retired whole: `appendix_seed` (kept verbatim as
   `sections/appendix_seed_v8_2026-09-19.tex`). Tables and figures cut where a proof, a table or the
   paragraph beside them already carried the content: the epoch/seed ladders, the scorer-scale grids,
   the vetting table, the two extra-pair head-to-heads, the TriviaQA certificate table,
   `tab:repairs`, the contamination table, five figures. The two-judge and two-pipeline paragraphs
   were **compressed, not dropped** --- they are concessions.
2. **The last three pages came from font size, not from deleting results.** The appendix body is now
   `\small` (floats are already `\footnotesize`, an absolute size, so no table or caption changed).
   That was worth 23 -> 20 pages on its own, against the alternative of deleting Proposition 4's only
   real test, the concession that only two anchors climb reproducibly, or the paper's own headline
   table. **Try the typographic lever before the scientific one** (AGENTS.md, caution (ar)).
3. **Read-through 6** found a caution-(aj) defect: two sentences, one in the main text, cited
   Appendix D for the ladders, which are in Appendix E. Re-pointed and guarded as a property.
4. `analysis/audit_numbers.py`: 2,188 numeric literals, **one** not in a CSV (`64256`, documented).

## Running now --- three classes locally, six arms queued on a second host

### Local box (4x A100, SHARED with another agent session --- see the OOM note)

| arm | class | card | state at 17:00 | ETA |
|---|---|---|---|---|
| feat-134 | creative 150x128 | GPU 4 | 7800/19200 (started 13:07) | ~12h |
| feat-134 | neutral 200x128 | GPU 4 | relaunched 16:48 under a supervisor | ~28h |
| feat-134 | factual 150x128 | GPU 1 | relaunched 16:48 under a supervisor | ~14h |

* **feat-134's neutral and factual classes were OOM-killed FIVE times** between 10:54 and 16:29, every
  time by **another Claude Code session running as the same Unix user** (`no_data_probe.py`, three
  processes at ~20 GB each plus one at 68 GB, under
  `/tmp/claude-1001/-...-agenticls-claude-only/...`). Our job holds 27.99 GB, so a card with under
  ~30 GB free kills it about an hour in. `nvidia-smi` shows those processes as ours, so **"free" now
  means free of other agent sessions too.**
* `scripts/run_comma7b128_supervise.sh <card_script> <gen_dir> <tag>` is the repair: it picks the
  emptiest eligible card (never GPU 3, never one another supervisor has claimed via
  `output/logs/claims/gpu<N>`), runs the card launcher **byte-identical**, and retries up to 12 times.
  It deliberately introduces **no** `--batch-size` and **no** allocator flag, because the arm rests on
  a bit-identity reward gate and cuBLAS can pick kernels by available workspace.
* **feat-134 CAN NEVER MOVE HOSTS.** Its reproduction gate demands ranks 0--63 of its reward cache be
  bit-identical to `results/selection_rewards64_comma7b.csv`, drawn on a local A100. Different silicon
  changes bf16 reduction order, hence sampled tokens, hence the rewards --- so the gate could never
  pass elsewhere and its own pre-registration forbids reading n=128 when it fails.
* `scripts/run_comma7b128_card2b.sh` (GPU 4) still owns the merge and the scoring, waiting on
  `GEN_DONE` in the neutral and factual directories, both written only on `rc=0`.
* **feat-135 stage 1 (vetting) PASSED** and stage 2 stays local **as registered** --- it is
  **running now on GPU 2**, started 17:20. The queue shell that was waiting on feat-134's factual
  `GEN_DONE` was killed by PID and replaced by `scripts/run_kl3m37b_breadth64_card.sh` under the same
  supervisor, because GPU 2 came free at 17:15 with 81 GB and the wait was only ever for a card. The
  arm is `scripts/run_breadth64.sh` unmodified either way, so the protocol is unchanged; it now runs
  on a card it does not share, which the pre-registration's co-residency note permits and improves on.
  ETA ~7 gpu-h.

### Second host (8x H100-80GB, idle, `~/v` only) --- feat-136

`results/onset_prediction_breadth_ladders.md` is committed with every band fixed. Six breadth arms
completing **two within-family capability ladders** (KL3M 170M/520M/1.7B/3.7B, Pleias
350M/1.2B/3B) plus the **one fixed-size data ablation the model set allows** (comma-1t vs comma-2t at
7B), and re-drawing TinyComma and Comma-7B to ask whether a paired difference survives a change of
**hardware** as feat-131/133 showed it survives a change of **seed**.

Environment is pinned to the local one exactly (python 3.12, torch 2.10.0+cu128, transformers 5.16.1,
and every other pin), verified by loading all seven models and running the suite there. Three
environment facts worth keeping:

* **`/tmp` is READ-ONLY on that host.** That is what broke `python3 -m venv` (no `ensurepip`, needs
  sudo) and two `uv` installs (`curl: (23)`, then `mktemp: Read-only file system`). The fix is
  `TMPDIR=$HOME/v/tmp`, exported by `~/v/env.sh`, which every remote command sources.
* **No secret is on that host.** `~/v/DA5001_Project/.env` is a single comment line; every model is in
  `hf_cache` and every run sets `HF_HUB_OFFLINE=1`, so no token is ever needed there.
* The suite there reports 170 failures, **all** of them the absent manuscript (`~/v/sub/satml/...`),
  absent local `output/` run directories, or one test that hardcodes `/tmp`. A green suite on that
  host is **not** evidence about the manuscript and must never be quoted as such.

## feat-136's instrument gate FAILED, and the gate was the thing that was wrong

This is the most important thing in this session to carry forward, and it is written up in full in
`results/onset_prediction_breadth_ladders.md`'s scoring log and as **caution (as)** in AGENTS.md.

G0a re-scored the committed arm's own `32,000` candidates on the second host with the input text held
identical, and required `99.9%` of rewards within `1e-2`. It read `17.6%`. The reward model is
**byte-identical** on both hosts, so the control that settled it removed hardware from the question
entirely --- same host, same weights, same texts, `--batch-size 8` against `16` --- and **that also
failed, at `60.0%`, with a larger maximum deviation than the cross-host run.** The registered
threshold was unsatisfiable by two runs on one machine. The false premise was ours, in the
pre-registration: *"bf16 reduction order moves it by ~1e-3"* is an fp32 figure.

* Withdrawn: `1e-2`, `99.9%`, `99%`. They stay in the source and the CSV marked `WITHDRAWN` so the
  record can be audited.
* Kept: the defect scale the same paragraph registered in advance, *"moves a reward by whole nats"*,
  read as `1.0` --- and a test asserts it stays `5x` above the measured value so it cannot drift
  toward the data.
* The repair was made while all three arms were still generating and no band existed. That is
  recorded, and so is the fact that the FAIL was read before the threshold was questioned.
* **The instrument question is answered by what was already registered**, not by a new gate: the two
  host-transfer arms' prediction `|D_hostB - D_local| <= 0.0610`, fixed in advance from feat-131.

**And the control is a result worth a paragraph of the paper.** The reward is reproducible to only
`~0.09` nats within one host; `40%` of rewards move by more than `0.01` on a batch-size change alone;
and **the completion best-of-`n` actually serves changes on `~3.0%` of (prompt, `n`) cells within one
host and `4.3%` across hosts.** The certificate is untouched --- Proposition 1 holds for ANY score and
ANY tie rule --- so **the guarantee is numerically robust exactly where the realisation is not.** It
had no committed bands, so when it lands in the paper it needs a `results/*_note.md` saying so, never
an `onset_prediction_*.md` (the pattern `results/selection_alpaca_note.md` already sets, which
`tests/test_preregistration_count.py` enforces). Two further controls are running to make it
actionable: fp32 against bf16 on one host, and fp32 batch 8 against fp32 batch 16 --- if the second
agrees tightly, **scoring in fp32 restores a reproducible served completion**, which is a concrete
recommendation rather than only a caveat.

## feat-137 --- the judge-free axis gets its second anchor

`results/onset_prediction_verifiable_comma1t.md`, bands committed before the run. The judge-free
axis (GSM8K exact match, no judge anywhere) rests on **one** anchor, and it is the axis that exists
to answer every objection to the judge. This adds `common-pile/comma-v0.1-1t` --- the same
architecture at the same 7B scale as Comma-7B, differing only in corpus size --- so the
training-data ablation runs on **both** axes at once, because feat-136's `comma1thb` is measuring
the judged half of the same comparison.

All **four** joint outcomes are given a reading in advance, including the two that damage the paper:
a judged CLIMB with no verifiable climb behind it is recorded as "a judged gain that no verifiable
improvement backs", and a judged SATURATION with a verifiable climb means the judged null is a judge
artefact and weakens every judged null the paper reports. **G0 is a floor gate** --- n=1 accuracy
must be at least 0.05 --- because a model that cannot do the task at all produces a flat curve, and
a flat curve from a floor is not a saturation.

## Scoring is already prepared --- both readings are fixed before the data lands

| arm | command | refuses cleanly with no data |
|---|---|---|
| feat-136 | `.venv/bin/python analysis/score_breadth_ladders.py --out results` | yes, "G0a ... NOT SCORED" |
| feat-137 | scorer not yet written; bands are in the log and the floor gate G0 comes first | --- |
| feat-134 | `.venv/bin/python analysis/score_n128.py --anchor comma7b --out results` | yes, "one of the two reward caches is missing" |
| feat-135 | `.venv/bin/python analysis/score_kl3m37b_breadth64.py --out results` | yes, "G2 coverage: FAIL ... NOT SCORED" |

`analysis/score_kl3m37b_breadth64.py` is new, because the breadth scorer on record gates each anchor
on reproducing a committed `n=8` arm and **this anchor has none by design**. It implements feat-135's
own gates (G2 coverage and G4 length block the band; G3 empties are reported and never gated) plus
the **MARGINAL rule at 2.0 interval half-widths**, and it reuses `selection_breadth.gate` and
`nonempty_gain` so the empty fraction is the committed definition rather than a lookalike.
`tests/test_kl3m37b_scorer.py` exercises all eight branches on synthetic CSVs and the scorer was
mutated three ways to prove they bite (caution (v)).

Two background waiters are armed and will wake this session:

* `scripts/wait_comma7b128.sh 3569258` --- reads only the log text after the last `START creative`,
  because a whole-file grep fired instantly on card2b's **first** attempt's abort (12:38) when a
  second invocation was already running (13:07). It also exits if the owner shell disappears without
  a completion line, checked with `kill -0`, never `pgrep` (caution (c)).
* a waiter on `output/logs/breadth64_kl3m37b.log` for `kl3m37b DONE` or a generation failure.

## Recommended next step

Check `output/logs/comma7b128_card2.log` for `COMMA-7B n=128 ARM DRAINED` and
`output/logs/breadth64_kl3m37b.log` for `kl3m37b DONE`. Then:

1. **Score feat-134** against the bands in `results/onset_prediction_comma7b_n128.md`. The sentence
   it lands in is in `sections/appendix_selection.tex`: "We did not take Comma-7B past $64$, so where
   the strongest anchor's ceiling sits is open."
2. **Score feat-135** against `results/onset_prediction_kl3m37b_breadth64.md`, whose verdict table is
   already written, including the **MARGINAL** rule at 2.0 interval half-widths. If it reads CLIMBS
   the appendix's two-anchor statement becomes three anchors in two families; if it saturates, the
   statement stays and gains the sentence that the family confound was tested.
3. Rebuild the artifact and re-run `analysis/compute_hours.py` once both arms close.

**Do not** compute either band before its gate passes (the rule both logs carry), and do not set a
single-order judged level from one sweep against another sweep's (caution (ap)).

---

## Update --- 2026-09-19 late evening

### The 70B positive control PASSED, so feat-136's G1 is interpretable

`unsloth/Meta-Llama-3.1-70B` in the risky slot, at the record's protocol byte for byte: **$0.480$ of
passages leaking and `max_recall` $0.9924$**, against a committed band of $\ge 0.20$ and $\ge 0.50$
and a record of $0.500$ / $1.0000$. Both halves PASS and every column lands within a re-draw. The
vetting pipeline's power on the second host is now **demonstrated rather than inherited**, which is
what a G1 pass needs: the same code detects a pre-training memoriser on nearly half of the same $50$
passages while Pleias-350M, KL3M-170M and KL3M-520M read exactly $0.0$. The OLMo-2-7B supplementary
control stays uninformative and is explicitly not what licenses G1.

### feat-136's first arm is INVALID, and the defect was in our own pre-registration

`tc18bhb` finished and **G0b failed**, $76.5$ words against the registered $72.2$. The cause is not
hardware: the empty fraction moved the wrong way to explain it ($0.018 \to 0.052$, which pushes a
mean DOWN) and the non-empty median moved $57 \to 74$. `run_breadth64.sh` self-pairs
(`--safe-model-path` and `--risky-model-path` get the same model); `sel_anchor64`, the directory the
$72.2$ came from, pairs the anchor with the 8B risky model and is not a breadth arm. The comparison
changed the host AND the pipeline, so per caution (w) the arm is **INVALID, not failed**, and its own
numbers are deliberately not read. **Confined to one arm** --- the gate touches only `role=host`
arms, and `comma7bhb`'s counterpart is a real breadth arm measuring $99.18$ against its registered
$99.2$. New caution **(at)**.

The scorer was repaired **before the replacement data existed**: a host arm now names the directory
it replicates, both the length reference and the local band are derived by the scorer's own code,
`pairing()` compares `target_model`/`anchor_model` across the two runs, and a structural failure is
no longer described as an undetermined length drift. Five guards, four mutations, each failing by
name.

### Also done

* `on_record()` now asserts three things, not one: the delta to `5e-4`, the MARGINAL **verdict**
  (the only thing caution (ap) lets the half-width ratio predict, so it blocks), and the ratio inside
  a bootstrap-seed slack guarded in **both** directions --- at least the widest measured ten-seed
  span, and narrower than the gap between the two closest committed ratios.
* `g5_ceiling()` and G0b's cross-check printing, both promised in the pre-registration and both
  previously unimplemented.
* feat-137's compute correction recorded **in feat-137's own log**, beneath the `## Scoring log`
  heading rather than by editing the estimate, which line 3 of that file promises is never touched.
  The registered $\approx 13$ gpu-hours should be $4$--$5$: `compute_hours.csv` already held
  `verifiable_comma7b` at $6.88$.

## Running now

State as read at **22:35** (clock read, not estimated):

| arm | where | state at 22:35 |
|---|---|---|
| feat-134 `neutral` | local GPU 4, pid 4044862 | generating since 16:48 |
| feat-134 `factual` | local GPU 1, pid 4046231 | generating since 16:48 |
| **feat-134 `creative`** | --- | **OOM-KILLED 22:28:44**, relaunched under the supervisor |
| feat-135 KL3M-3.7B | local GPU 2, pid 4134532 | generating |
| **feat-136 `tc18bsp`** (the missing local counterpart) | local GPU 2 | **OOM-KILLED 20:41:22** after 29 min (started 20:12:22), relaunched |
| feat-136 `comma7bhb`, `comma1thb`, `pleias350mhb`, `kl3m170mhb`, `kl3m520mhb` | host B | 14--39% |
| feat-137 `verifiable_comma1t` | host B | generating |

`tc18bsp` is `run_breadth64.sh <gpu> jacquelinehe/tinycomma-1.8b-llama3-tokenizer tc18bsp` --- the
same script and flags as the host arm, which is the whole point of it. It must stay **local**: it
exists to be the local side of a host-transfer comparison.

### The local box is contended by another agent session, and `h1.py` makes that expensive

Two arms died between 20:41 and 22:28, both to a **different Claude Code session** running as the
same Unix user out of `~/agenticls` (`keyblind_operators.py`, `learned_probe_nonlinear.py`). At 22:35
it holds $40.0$ GB on GPU 0 and $24.7$ GB on GPU 1, and the two jobs that did the killing were
$54.9$ GB (GPU 2, took `tc18bsp`) and $22.1$ GB (GPU 4, took `creative`). That is the **sixth and
seventh** such kill today.

**What makes it expensive is ours, not theirs: `h1.py` writes a class's trajectories only when the
class finishes.** All three feat-134 class directories hold nothing but zero-byte files for the
classes they are not generating, so `creative` lost **12.5 hours** (`09:59` to `22:28`) and left no
partial output at all. `neutral` and `factual` are each ~6 h in with the same exposure. Do **not**
patch `h1.py` to flush incrementally while three of its processes are running (never edit a script
while it runs). The mitigation available now is the supervisor, which waits for a card with
`MIN_FREE_MIB=34000` free and retries.

## Recommended next step

Score `comma7bhb` when it lands (it has a valid counterpart), then `tc18bhb` once `tc18bsp`
finishes. Do not fold any feat-136 number into the manuscript until its arm passes G0a, G0b and G2.


---

## Update --- 2026-09-20: host B finished, and all eight of its H100s were refilled

### What host B returned

All six `feat-136` arms and `feat-137` exited `rc=0`. **Four anchors scored.** Comma-7B (1T)
**CLIMBS** $+0.0970\ [+0.0560, +0.1390]$ at $2.34$ half-widths; Pleias-350M, KL3M-170M and
KL3M-520M all read **SATURATED BY 8** and all are MARGINAL. G5 clears every one, so the nulls are
nulls rather than a bounded metric near its limit.

* **H1 (capability) is falsified** in KL3M and Pleias, upheld in Comma --- reported both ways.
* **H2 (family) SURVIVES:** no non-Comma anchor clears $2.0$ half-widths, over six anchors in two
  families from $0.168$B to $3$B.
* **H3 (training data) is answered:** at fixed $7$B the 1T checkpoint climbs $+0.0970$ against 2T's
  $+0.1010$ --- a gap of $0.004$, inside the $0.013$ caution (ap) calls the honest scale.
* `feat-137`: majority vote $+0.1740\ [+0.1340, +0.2160]$ CLIMBS, pointwise reward $+0.0600$
  CLIMBS; cross-axis reading is judged CLIMBS $\times$ judge-free CLIMBS.

**`comma7bhb` is blocked by G0b at $+5.3\%$ and its band has NOT been computed or looked at.** The
cause decomposes exactly --- pipelines match, and the whole shift is the empty rate
($0.094 \to 0.020$) while length given non-empty moved $-2.7\%$. A gate is not repaired after its
verdict is read, so the repaired statistic is registered for NEW arms only (see I1 below).

### Three new pre-registrations, all committed BEFORE their arms started

| | question | arms | state |
|---|---|---|---|
| **feat-138** | do the ladder's verdicts survive a fresh draw? | 4 seed replications, host B GPUs 0--3 | generating |
| **feat-140** | the whole non-Comma ladder on ONE host | Pleias-1.2B/3B, KL3M-1.7B/3.7B, GPUs 4--7 | generating |
| **feat-139** | is the judge-free climb Comma-specific too? | 4 $n=1$ floor probes, queued behind GPUs 4--7 | waiting |

Their scoring logs, unscored as of this writing:
`results/onset_prediction_breadth_seed_replication.md` (feat-138),
`results/onset_prediction_single_host_ladder.md` (feat-140),
`results/onset_prediction_judgefree_offcomma.md` (feat-139).

`feat-138` exists because `feat-136`'s own rule prints MARGINAL for its three nulls and says they
are not promoted without a replication --- so the nulls carrying H2 could not be used until re-drawn.
Exactly one flag changes (`--seeds 42 43 44 -> 52 53 54`), checked mechanically; `--batch-size 32`
is held, which is `feat-132`'s whole lesson.

`feat-140` exists because H2's six anchors were split across two machines and caution (at) showed
that is a real confound. The four missing anchors were downloaded and verified to load offline with
embedding rows matching the tokenizer.

`feat-139` probes $n=1$ first because four full ladders the floor then refuses would cost
$\approx 20$ gpu-hours to learn what $n=1$ answers in minutes. **Its committed prediction is that
all four fall below the floor**, which would mean H2 is a judged-axis finding and must be scoped
that way in the paper.

### Scoring machinery, written and mutation-tested before any of this data exists

* `analysis/arm_rank0_stats.py` reduces an arm's $2.5$ GB of generations to one row, because I1
  needs `mean_words_nonempty` and the empty rate as data. It asserts the identity
  `mean = (1 - empty) * nonempty`.
* `analysis/score_seed_and_ladder.py` scores both pre-registrations, importing every constant from
  `score_breadth_ladders` rather than retyping it. Six mutations, six named failures.
* **I1 is the repaired statistic** --- length GIVEN NON-EMPTY --- pinned in both directions so it is
  not a weakening: an arm whose raw mean moves $11\%$ with non-empty length identical must PASS, and
  one whose non-empty length moves $25\%$ behind a compensating empty rate must FAIL.

### A manuscript sentence was falsified by feat-137 and has been scoped

Section 5 said, unscoped, *"four draws of it beat all $28$ reward cells on either task"*. Those $28$
are four **scorers** at the $7$B anchor; at the **1T** anchor four draws gain $+0.050$ against its
scorer's best $+0.060$ and it takes **eight**. The sentence now says
*"...at this anchor, eight at its 1T sibling"*. Recompiled: exit $0$, $0$ overfull, $0$ `??`, zero
body lines on page 10. Two guards, one scoped to its arm and one new and guarded both ways.

### Local box

`feat-134` `neutral`/`factual` generating, `creative` restarted under the supervisor after an
OOM kill; `feat-135` generating; `feat-136`'s `tc18bsp` counterpart queued under the supervisor
behind the four claimed cards.

---

## Results in, 2026-09-20 ~04:00

### feat-138 replicates 4 of 4; feat-140's prediction fails 3 of 3; the PAIR is the finding

| what changed | moves observed | verdicts flipped |
|---|---|---|
| the seed, same host (`feat-138`) | $0.007$--$0.018$ | **0 of 4** |
| the host, same seeds (`feat-140`) | $0.018$--$0.070$ | **2 of 3** |

Pleias-3B moved $0.0700$, **beyond** the $0.0610$ that bounds every seed replication on record.
**A host change is not a re-draw.** Pleias-1.2B ($+0.0560$), Pleias-3B ($+0.0730$) and KL3M-1.7B
($+0.0470$) were all registered SATURATED and all three CLIMB on host B; I1/I3/I4 pass on each, so
these are like-for-like and not caution (at)'s defect.

**H2 survives its committed rule** --- no non-Comma anchor clears $2.0$ half-widths, largest $1.80$
--- **but "non-Comma anchors do not climb" is FALSE and the finding is restated: what separates the
families is STABILITY, not direction.** Comma sits at $2.07$--$2.43$ and reproduces; every non-Comma
reading sits below $1.81$ and moves enough to flip itself.

**Stated limit:** both Comma host-transfer arms are blocked by G0b, so there is **no** unblocked
cross-host Comma comparison and "Comma is stable across hosts" is unsupported.

KL3M-1.7B is at two of three draws saying CLIMBS. Its pre-registration said in advance that this
leaves its status genuinely open, and that is not revised.

### feat-139 stage 1: three of four sit 3--4x below the judge-free floor

$0.0140$, $0.0180$, $0.0160$ against `FLOOR = 0.05` and Comma-7B's $0.320$; flat across
$1.2$--$3.0$B. **The conclusion is not declared until KL3M-3.7B is measured** --- the registration
says four. Stage 2 unstarted for every anchor, as registered.

### Still running

* host B: `kl3m37bhb` (the last ladder arm) and the `kl3m37b` floor probe queued behind it.
* local: `feat-135` at $69\%$; `feat-134` `neutral`/`factual`/`creative` all restarted by their
  supervisors after OOM kills from the other agent session; `tc18bsp` **SUCCEEDED**.

### What is NOT resolved

`tc18bhb` fails G0b at $+6.1\%$ even against its proper counterpart `tc18bsp`, and on the repaired
statistic (length given non-empty) at $+6.5\%$. That is a real cross-host length shift at TinyComma,
not the pipeline artefact --- the pipeline artefact was the OLD reference. `comma7bhb` fails at
$+5.3\%$ whose whole content is the empty rate. **Neither band has been computed or looked at.**

---

## 2026-09-20 ~04:30 --- feat-140 completes at 8 of 8, and feat-141 is launched to attack its own conclusion

**KL3M-3.7B on host B** read $+0.0200\ [-0.0160, +0.0560]$, $0.56$ half-widths, SATURATED BY 8 --- no
verdict had been predicted for it (P4). It is nonetheless **NOT SCORED**, because its local
reference turned out to be an unfinished arm; see below.

**A defect in my own scoring run, caught and fixed structurally.** `h1.py` writes a class only when
that class completes, so `feat-135`'s still-generating local KL3M-3.7B arm yielded a stats row over
**350** prompts rather than $500$ --- its `creative` class had not been written --- and I1 compared
a mean over $350$ prompts against a mean over $500$ and **passed it at $+2.6\%$**. `I1` now refuses
unequal counts and refuses equal-but-short counts, and says *"one side is not the finished
500-prompt arm"* rather than describing it as a length disagreement. Three mutations, three named
failures.

**`results/onset_prediction_within_host_spread.md` (feat-141), committed before its arms started**,
exists to break `feat-140`'s own conclusion. That conclusion --- *a host change is not a re-draw* ---
rests on four within-host seed pairs, **none at an anchor that flipped**, and the three that flipped
have one host-B draw each, so their own spread is unmeasured. P1 is written so it can refute:
**if any within-host pair at Pleias-3B moves by $\ge 0.0700$, the conclusion is withdrawn in the
same words it was written.** P3 puts the stable anchor at risk too: Comma-7B (1T) must read CLIMBS
a third time at or above $2.0$ half-widths.

Eight arms are running on host B's eight cards, two further disjoint seed triples (`62 63 64`,
`72 73 74`) at each flipped anchor plus one each at KL3M-3.7B and Comma-7B (1T).
`scripts/run_breadth64_seed.sh` now takes the triple as an optional fourth argument **defaulting to
`52 53 54`**, so every `feat-138` invocation on record is byte-identical in behaviour.

### feat-139 stage 1, three of four measured

$0.0140$, $0.0180$, $0.0160$ against `FLOOR = 0.05`; KL3M-3.7B's probe is running now. The
registered conclusion is still not declared --- the registration says four.
