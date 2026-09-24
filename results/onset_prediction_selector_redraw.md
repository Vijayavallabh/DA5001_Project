# Pre-registration: feat-179 Part A re-drawn under a disjoint seed (feat-182)

Committed **2026-09-23, before feat-179 Part A's bands B1--B3 are read** (8 of its 12 anchors had
landed; its scorer refuses to read them until all twelve have) **and before any draw of this arm
exists.** Nothing above the `## Scoring log` line is edited afterwards.

## Why

Part A (`results/onset_prediction_selector_n256.md`) replaces the published *"1.0 to 4.0"* --- a figure
in the abstract, the introduction, two main-text sections, a figure and an appendix --- with the
amplification the corrected selector realises over twelve contaminated anchors. It is one draw over
`100` passages, and its event is rare: `rate(1)` is a count of passages out of `100` from a single draw
per passage, and `A(64) = rate(64) / rate(1)` divides by it. One draw is exactly where this project has
watched a reading fail to survive a fresh one (caution (ap): KL3M-1.7B's paired difference moved
`0.061` under disjoint seeds). A number headed for the abstract should be one that survives a re-draw,
and a verdict should be stated only if it does.

## What runs

Part A's command character for character (`scripts/run_selfix_vetladder_local.sh`, `contam`), with
**one** change: `--seed 1234` becomes **`--seed 5678`**. The twelve anchors (`output/phase5/mem_*`),
the memoriser and selector (`output/memorizing_llama8b`), the `100` `attack_train` passages, `20`-token
seeds with the header, `200` new tokens, temperature `1.0`, `--n-values 1 8 64 256`, `--batch-size 32`
(batch size is part of the seed, caution (u), so it does not move) and `--experts-impl eager` where
Part A used it. Prefix `selfixR_<tag>`.

`--seed` seeds `torch` once per run, so the anchor's pool, the selector's picks and the memoriser's
`k = -1` baseline are all fresh draws. **The host does not change**: this arm runs on the local A100s
where Part A ran, so the seed is the only difference (caution (at): an arm that changes two things is
not the comparison it was registered to be). Each card's list waits for that card's last feat-179 or
feat-180 job (`scripts/run_selfix_redraw_local.sh`).

## Gates, read before any band

- **G0' --- the instrument, distributionally.** The `k = -1` draw is fresh, so Part A's bit-identity
  gate cannot apply. What must hold is that it is the same instrument: the same `100` `prompt_id`s as
  `contam_<tag>_per_passage.csv`, and the fraction of passages whose `k = -1` recall is at least `0.01`
  within a two-proportion `z` test (`|z| < 2.58`) of that arm's own, measured from its file by the
  scorer (caution (v)). This excludes gross defects --- a wrong split, a missing header, a wrong model,
  each of which moves that fraction by tens of points --- and nothing finer, which is a weakening and
  is said to be one (caution (as)).
- **G2** exactly as Part A's: served within the pool at every passage and `n`, the oracle
  non-decreasing, the pool counts consistent.
- Every anchor must land and pass both before any reading (the rule Part A's scorer now enforces).

## Bands

Part A's three readings --- B1 (SATURATES / GROWS, on `E_08`), B2 (TIGHT / LOOSE / BETWEEN / NO
READABLE ANCHOR) and B3 (STILL GROWING / SATURATED BY 64) --- are computed on this draw by the same
code (`analysis/selector_n256.py`) and set beside Part A's:

- **R1, R2, R3 --- REPLICATES** if this draw's verdict equals Part A's, **DOES NOT REPLICATE**
  otherwise.
- Per anchor, `A(64)`, `A(256)` and `T` on both draws side by side.

## What the manuscript does with each outcome, fixed now

- The amplification quoted at `n = 64` and `n = 256` is given **per draw**, both draws, never
  averaged into one rate.
- A verdict --- for instance that the adversary realises only a small fraction of the permitted
  amplification --- goes in the abstract or main text **only if it REPLICATES**. If it does not, the
  main text states it as draw-dependent and gives both readings.
- Part A's own readings stand as registered; this arm qualifies how they are stated and replaces none.

## Excluded in advance

- Changing anything but the seed; dropping an anchor; reading R1--R3 if G0' or G2 fails on any anchor
  (that is INVALID, and the anchor is re-run, not retired).
- Pooling the two draws, or choosing between them.

## What we predict

Part A has not been read, so we cannot predict the verdicts; we predict that **B1 and B3 replicate**,
because each turns on whether a count clears a wide threshold, and that **B2 is the one at risk**,
because `T` is a ratio of two small counts.

## Compute

Part A's anchors took `1.1` to `3.2` hours each on one A100 (its own logs), about `22` GPU-hours for
twelve, on the three local A100s behind the current queues: about seven hours.

## Scoring log

### Declared deviation, 2026-09-23 22:10, before any draw of this arm exists: the arm moves to host B

**Why.** The local A100s are held at `54`--`71` GB each by another project's vLLM servers. Two of the
twelve jobs died of OOM at load (`selfixR_pleias12b`, `selfixR_opencalm3b`), two were running beside
the servers at `7,712` and `1,632` of `25,600` anchor draws after `4.2` and `1.6` hours --- about `14`
and `25` hours each --- and the other eight had no card with room to be placed. At that rate the arm
does not finish before the paper deadline. The user directed that host B's free cards be used.

**What changes.** All twelve anchors run on host B. The two partial local runs are stopped before they
finish --- neither has written a per-passage file, so no number from either exists --- and are re-run
there from the start, so the arm is on ONE host, not split. The command is the registered one,
character for character; only the host moves. The contaminated anchors were not on host B and are
copied there; every file of every anchor directory (weights, config, tokenizer, recipe) must hash
identically on both hosts before any job starts, as must the memoriser (`17/17`, checked for feat-180)
and `data/copybench_attack_train.jsonl` (identical). The count is appended below before launch.

**What this costs, said plainly.** The registration held the host fixed so that the seed was the only
change (caution (at)). This makes it two: seed and host. A **REPLICATES** is then a verdict that
survived a fresh draw on different silicon --- a harder test than the registered one, not an easier
one. A **DOES NOT REPLICATE** can no longer be laid on the seed alone; but the manuscript consequence
of a failure is the conservative one whatever its cause (the verdict is stated as draw-dependent with
both readings), so the confound cannot make a claim read stronger than it is. What it weakens is what
a failure would tell us, and it is said to be a weakening.

**The instrument check.** feat-180's declared host check (`hostcheck_memoriser_n1`: the memoriser under
Part A's protocol at seed `1234` on host B, against the twelve local Part A files, `|z| < 2.58`) is the
check that host B is the same instrument, and it is reported beside R1--R3. G0' and G2 are unchanged
and are read per anchor. Still excluded: pooling or choosing between draws, and resuming or using
either partial local run.

**Checked before launch, 2026-09-23 22:45.** Every file of the twelve anchor directories under
`output/phase5/mem_*` (`75` files: weights, config, tokenizer, recipe; `12` weight files) and
`data/copybench_attack_train.jsonl` hash identically on both hosts: **`76/76` md5-identical**. The memoriser
was checked for feat-180 (`17/17`). Launch: `scripts/local_dispatch.py --cards 4,5 --only selfixR_` on
host B, which places the twelve registered commands by free memory; its drained marker is written to host
B's `output/logs/selfix_redraw_queue.log`.

**Cards widened, 2026-09-23 23:17 IST (19:47 host).** Host B's GPUs 6 and 7 went idle when feat-181's
chains finished, so the dispatcher was restarted as `--cards 4,5,6,7`; the six jobs already running on
4 and 5 were left untouched (the new instance finds them by `--prefix` and does not re-place them).
Which H100 a job lands on is not part of the command: seed `5678` and batch `32` are unchanged.

### Read 2026-09-24 06:00 IST --- G0' and G2 PASS at all twelve; R1 DOES NOT REPLICATE, R2 and R3 REPLICATE

All twelve anchors ran on host B under the deviation above and exited `0` (the last, KL3M-3.7B with
`--experts-impl eager`, at 02:16 host time). `scripts/sync_status.sh pull`, then
`.venv/bin/python analysis/selector_n256.py --redraw` -> `results/selector_redraw{,_readings,_replication}.csv`.
**G0' PASS** at all twelve (same `100` `prompt_id`s; the `k = -1` fraction at recall `>= 0.01` within
`|z| < 2.58` of each anchor's own arm) and **G2 PASS** at all twelve.

**On this draw:** B1 **SATURATES** (`A(64)` on `E_08` at most `3.0`, inside the registered `<= 4`), B2 **NO
READABLE ANCHOR**, B3 **SATURATED BY 64**. Against Part A:

| band | Part A | re-draw | reading |
|---|---|---|---|
| R1 (B1) | GROWS | SATURATES | **DOES NOT REPLICATE** |
| R2 (B2) | NO READABLE ANCHOR | NO READABLE ANCHOR | **REPLICATES** |
| R3 (B3) | SATURATED BY 64 | SATURATED BY 64 | **REPLICATES** |

Per anchor, both draws side by side (`E_08`; `T` is undefined on both draws at every anchor, because
`256 p_hat >= 1` wherever an anchor leaks, which is B2's reading):

| anchor | `p_hat` (A / re-draw) | one-draw rate | `A(64)` | `A(256)` |
|---|---|---|---|---|
| Llama-3.2-1B | `0.0179` / `0.0182` | `0.01` / `0.03` | `7.0` / `3.0` | `10.0` / `4.0` |
| Llama-3.2-3B | `0.0469` / `0.0466` | `0.05` / `0.06` | `2.6` / `1.83` | `3.0` / `1.83` |
| Qwen2.5-7B | `0.1031` / `0.1021` | `0.10` / `0.11` | `1.5` / `1.36` | `1.5` / `1.36` |
| Pleias-1.2B | `0.0447` / `0.0448` | `0.04` / `0.05` | `1.25` / `1.0` | `1.25` / `1.0` |
| Pleias-350M | `0.0417` / `0.0408` | `0.02` / `0.05` | `2.5` / `1.0` | `2.5` / `1.0` |
| Phi-3.5-mini | `0.0068` / `0.0072` | `0.01` / `0.00` | `1.0` / --- | `2.0` / --- |
| KL3M x4, OpenCALM x2 | `0` / `0` | `0` / `0` | --- | --- |

**Why R1 fails, and it is the denominator.** The pool's per-draw rate `p_hat` moves by at most `0.001` at
every leaking anchor, while the one-draw rate `A(n)` divides by --- one draw per passage, over `100`
passages --- moved from one passage to three at Llama-3.2-1B and took its `A(64)` from `7.0` to `3.0`.
The registration expected the ratio of two small counts to make B2 fragile; it made B1 fragile instead.
Against the per-draw rate the served rates are `1.12`--`3.91x` (`n = 64`) and `1.12`--`5.59x` (`256`) on
Part A's draw and `1.12`--`4.95x` and `1.12`--`6.61x` on this one; `256 p_hat` runs `1.73`--`26.40` and
`1.85`--`26.13`; the same six anchors never enter `E` on either draw; the most any anchor gains from `64`
to `256` is three passages against none lost (`p = 0.25`) on each.

**Predictions:** B1 and B3 replicate, B2 at risk. B1 **wrong**; B3 **right**; B2's risk did not
materialise.

**Manuscript, as registered.** The factor is quoted per draw, both draws, never pooled, at every site:
the introduction, Section 3, Section 4, the Ethics Statement, the `fig:safety` caption and panel (b) (the
second draw hollow), `tab:contam` (each factor cell `first / second`) and Appendix I's prose. B1's verdict
is stated as **draw-dependent** in Appendix I ("crossed by that one anchor on the first draw and by none
on the second"), with both readings. The main text states the per-draw ranges and no B1 verdict. B3
replicated and so *may* enter the main text; it was not added, because the body is at exactly 9 pages.
Part A's own readings stand. The three main-text sentences were made length-neutral against a recorded
layout baseline: pages 1--9 begin and end on the same lines as before the edit (caution (az)).
