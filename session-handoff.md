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
* **feat-135 stage 1 (vetting) PASSED** and stage 2 stays local **as registered**, queued in
  `scripts/run_kl3m37b_breadth64_queued.sh` behind the factual class's `GEN_DONE`.

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

## Scoring is already prepared --- both readings are fixed before the data lands

| arm | command | refuses cleanly with no data |
|---|---|---|
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
