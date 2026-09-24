# Pre-registration: both selection ladders to n = 512 (feat-181)

Committed **2026-09-23, before any draw beyond rank 255 exists**. Nothing above the `## Scoring log`
line is edited afterwards.

## Why

feat-172 (`results/onset_prediction_offsupport_ladder.md`) took the audited anchor to `n = 256` on two
workloads and read **STILL CLIMBING** on both --- `g(256) - g(128) = +0.0280 [+0.0025, +0.0553]` on
AlpacaEval and `+0.0306 [+0.0047, +0.0547]` on our `850` prompts --- but at `1.06` and `1.22`
half-widths, below the `1.71` at which a paired difference has already failed to replicate here
(caution (ap)). Measured from `n = 64` the two doublings contain zero on both workloads (post hoc), so
Appendix I now says it claims **neither a ceiling nor a slope**. And selection still trails the binding
meter off-support at `256`: `D3 = -0.0239 [-0.0429, -0.0050]`.

That registration named the next step and deferred it: *"If the answer is 'it would cross at 512',
that is a **separate** registration with its own bands."* This is that registration. One more doubling
answers both open questions with the same draws: does the gain keep rising past `n = 64`, and does
selection ever catch a binding meter off its anchor's support?

## What runs

**Host B, where both `n = 256` pools and their reward caches were made** --- a bit-identity gate is a
constraint on the silicon (feat-172's own scoring log; cautions (as), (at)). Every flag of the
committed launchers is kept: `scripts/run_offsupport.sh` for Arm A (AlpacaEval, `805` prompts through
the factual slot, `data/bench/alpaca`) and `scripts/run_onsupport.sh` for Arm B (our `850`: the `small`
shard `neutral`+`creative` and the `factual` shard, one process each, as the committed pool was drawn),
`--k-values 0.0 --max-new-tokens 200 --batch-size 64`, default `--seeds`.

**Only trajectory indices `256`--`511` are drawn**, with the new `--trajectory-start` flag
(`dap/e1.py`). Every batch is seeded by its own trajectory seed and batched over every prompt of its
process, so a tail run gives exactly the draws a full `512`-draw run would (`tests/test_seeds.py`
pins the seeds, the ids and the per-seed batches). The indices are split across cards in contiguous
ranges; which range a card draws does not change any draw.

Each class's committed `256`-draw file and its new shards are concatenated into one merged directory
per arm. `load_candidates` sorts by seed, whose low bits are the trajectory index, so rank `j` is
trajectory `j` and `n = 512` nests over the committed arms. Then the reward pass
(`analysis/selection_scaling.py --max-n 512 --rewards-only`, batch `16` as committed), then judge~B over
the grid `1 ... 512`, then for Arm A `analysis/order_averaged_h2h.py` at `--n 512`, the committed
AlpacaEval invocation with only `--n`, `--tag` and `--rewards` changed.

## Gates, read before any band

- **G0 --- the new code path is the committed pipeline.** Trajectory `255` is regenerated with
  `--trajectory-start 255 --trajectories-per-prompt 1` in each of the three processes (Arm A; Arm B
  `small`; Arm B `factual`) and its `generation` must be **byte-identical** to the committed
  trajectory-`255` draw on every prompt (`805`, `350`, `500`). A failure is **INVALID**: no extension
  draw is used.
- **G1 --- the rewards.** Ranks `0`--`255` of each `n = 512` reward cache are bit-identical to
  `results/offsup_rewards256.csv` (`206,080` floats) and `results/onsup_rewards256.csv` (`217,600`),
  compared with `==`. `512`, `256` and `16` make ranks `0`--`255` of every prompt fall in the same batch
  tuples, the argument feat-172's gate passed on. A failure is **INAPPLICABLE**.
- **G2 --- the merged pool.** Every prompt has exactly `512` candidates, trajectory ids `0`--`511` once
  each.

## Bands, committed before the data

Judge~B, paired within the `n = 512` pass (a judged level is never compared across passes, caution (ap)).

- **C2 (Arm A and Arm B) --- the last doubling.** Paired `g(512) - g(256)`: **STILL CLIMBING** if its
  interval excludes zero above, **SATURATED** if it contains zero, **TURNS OVER** if it excludes zero
  below.
- **C3 (Arm A and Arm B) --- the question Appendix I leaves open.** Paired `g(512) - g(64)`: **CLIMBS
  PAST 64** if its interval excludes zero above, **NOT RESOLVED** if it contains zero, **FALLS** if it
  excludes zero below.
- **Marginality, registered rather than added afterwards.** Every C2 and C3 reading carries its point
  estimate over its half-width; below `1.7` it is labelled **MARGINAL** (caution (ap)'s measured
  boundary), and a MARGINAL reading is reported as a label, never as a measured slope.
- **C1 (Arm A) --- does selection catch the binding meter off-support?** Order-averaged `D3` at
  `n = 512` against the `k = 1` metered arm, the committed opponent and judge: **CATCHES** if
  `D3 >= 0`; **STILL BEHIND** if its interval excludes zero below; **UNRESOLVED** otherwise. Beside it,
  the order-averaged `g(512) - g(256)` and `g(512) - g(64)` from the per-prompt files, which draw
  nothing and so pair exactly across passes (asserted: `u_sel_n1` identical on every prompt).
- **C4 --- the price.** `log 512 = 6.238` nats against the meter's realised spend on AlpacaEval
  (`145.10`, feat-172's B3, read from that arm's own trajectories), as a ratio.

## Excluded in advance

- Drawing past `512` in this arm, whatever C2 reads; that would be another registration.
- Reading any band if G0, G1 or G2 fails.
- Comparing a judged level from this pass with one from any other.
- Presenting CATCHES as rescuing the headline: the headline is a claim at `n = 64`, and feat-172's
  rule stands --- the scoping concession is rewritten, not deleted.
- Pooling this pass with feat-172's, or reading its `n <= 256` rungs as a replication of them: they
  are the same draws re-judged under a larger grid.

## What we predict

**C2 SATURATED on both arms**: the ladders have gained about `+0.015` per doubling from `8` to `256`,
against a paired half-width near `0.027`. **C3 CLIMBS PAST 64 on both, MARGINAL**: three doublings at
that rate is about `+0.045` against a half-width near `0.03`. **C1 STILL BEHIND**: the order-averaged
selection gain rose `+0.0100` over the two doublings from `64` to `256` and the deficit is `0.0239`.

## What the manuscript does with each outcome, fixed now

- **C3 CLIMBS PAST 64** (either arm): Appendix I's *"we claim neither a ceiling nor a slope"* is
  replaced by the measured `g(512) - g(64)` for each arm, with its MARGINAL label if it has one.
  **NOT RESOLVED** on both: the sentence stands and is extended to `512`. **Mixed**: each arm is
  reported as it read, not pooled.
- **C1 CATCHES**: the support-ceiling paragraph states the crossing `n` and its certificate against
  `145.1`, and the main text's *"as does moving off prefix completion"* is scoped to `n <= 256`.
  **STILL BEHIND** or **UNRESOLVED**: *"fails off-support at 256 as at 64"* is extended to `512` with
  the reading's own word.
- Nothing in the main text changes unless C1 reads CATCHES.

## Compute

From feat-172's own logs on host B: Arm A's `256` draws took `14.4` card-hours, Arm B's `small` shard
`4.4` and its `factual` shard about `10`. The extension is the same again, about `29` GPU-hours,
split over host B's eight idle H100s (about `4.5` hours wall-clock); rewards about `2.2`, judging and
the head-to-head about `4`. About `35` GPU-hours in all, `17` to `18` per arm. **Run at the user's
instruction of 2026-09-23 to use host B's free cards**, which were all idle when this was written.

## Scoring log

### G0, read 2026-09-23 before any extension draw --- PASS on all three processes

Trajectory `255` regenerated through `--trajectory-start 255 --trajectories-per-prompt 1`
(`analysis/n512_pool.py g0`) is byte-identical in `generation` to the committed draw on **805/805**
AlpacaEval prompts, **200/200** `neutral`, **150/150** `creative` and **500/500** `factual` --- the
new code path is the committed pipeline on this host. One of the three ran on a card shared with
another project's vLLM server, which started there minutes before the launch; the draw is identical
regardless.

**Launch changed from fixed cards to placement by free memory, before any extension draw.** Host B
was not idle for long: within twenty minutes of the check above, another project's vLLM servers
took GPUs 0--3 and moved between cards twice. Our processes peak at `21`--`30` GB, so
`scripts/n512_dispatch.py` places each job on whichever card has at least `36` GB free, at most two
of ours per card, and re-places a job whose process fails. Placement cannot change a draw: G0 above
was produced on three different cards, one of them shared, and matched byte for byte. The index
ranges are unchanged in union (`256`--`511`), cut finer (`32` for Arm A, `42`--`43` for `factual`,
`128` for `small`) so that a lost job costs less.

### Read 2026-09-23 23:15 --- G1, G2 PASS; C2 SATURATED on both; C3 NOT RESOLVED on both; C1 UNRESOLVED

All sixteen extension jobs ended `done` (four after one or two re-placements for OOM beside other
processes; placement cannot change a draw, as G0 showed). `scripts/run_n512_post.sh a 6` and `b 7` on
host B: **G2 PASS** in the merge (`805` factual; `200` neutral, `150` creative, `500` factual: every prompt
holds trajectory ids `0`--`511` exactly once), **G1 PASS** on both (ranks `0`--`255` of each `n = 512`
reward cache `==` the committed `n = 256` cache, `206,080` and `217,600` floats), then judge~B over the
grid to `512` and Arm A's order-averaged head-to-head at `n = 512`. Pulled with
`scripts/sync_status.sh pull`; `.venv/bin/python analysis/score_n512.py` -> `results/n512_ladder.csv`
(G1 and G2' re-read locally: PASS, PASS).

| band | arm | quantity | value [95%] | half-widths | reading |
|---|---|---|---|---|---|
| C2 | A (AlpacaEval) | `g(512) - g(256)` | `+0.0056 [-0.0193, +0.0311]` | `0.22` | **SATURATED** |
| C2 | B (our 850) | `g(512) - g(256)` | `+0.0071 [-0.0165, +0.0300]` | `0.30` | **SATURATED** |
| C3 | A | `g(512) - g(64)` | `+0.0311 [-0.0025, +0.0634]` | `0.94` | **NOT RESOLVED** |
| C3 | B | `g(512) - g(64)` | `+0.0312 [-0.0018, +0.0635]` | `0.95` | **NOT RESOLVED** |
| C1 | A | order-averaged `D3` at `n = 512` | `-0.0053 [-0.0248, +0.0143]` | | **UNRESOLVED** |
| C1 | A | order-averaged `g(512) - g(256)` | `+0.0186 [+0.0065, +0.0301]` | `1.58` | (beside C1, no band) |
| C1 | A | order-averaged `g(512) - g(64)` | `+0.0286 [+0.0115, +0.0438]` | `1.77` | (beside C1, no band) |
| C4 | A | `log 512` against the meter's realised `145.10` | `6.238` nats | | ratio `23.3` |

No C2 or C3 reading is MARGINAL, because none is directional: MARGINAL labels a reading that excludes
zero at under `1.7` half-widths. The two order-averaged differences exclude zero at `1.58` and `1.77`
half-widths --- one below and one just above the boundary where a paired difference has already failed
to replicate here (caution (ap)) --- and they carry no band, so they are reported beside C1 and read
as nothing.

**Predictions:** C2 SATURATED on both, **right**; C3 CLIMBS PAST 64 on both, MARGINAL, **wrong** (NOT
RESOLVED on both); C1 STILL BEHIND, **wrong** (UNRESOLVED: the deficit narrowed to `-0.0053`, and its
interval now contains zero).

**Manuscript, as registered.** C3 NOT RESOLVED on both: Appendix I's *"we claim neither a ceiling nor a
slope"* stands and is extended to `512`, with the last doubling, the three-doubling totals and the
order-averaged pair beside them at their half-widths. C1 UNRESOLVED: *"the reversal fails off-support at
`256` as at `64`"* stands (D3 at `256` is still clear of zero) and is extended to `512` with the reading's
own word, *unresolved*, and C4's `log 512 = 6.24` nats, `23.3x` below the meter's spend. Nothing in the
main text changes: C1 did not read CATCHES, and no main-text sentence states the off-support deficit.
Limitations' "judged arms reach `n = 256`" and "to `n = 256` no judged ladder turns over" become `512`
(C2 reads SATURATED, not TURNS OVER, on both).
