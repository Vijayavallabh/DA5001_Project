# Pre-registration: the compute-matched arm, measured instead of proxied (feat-162)

Committed **before any timing exists on host B**. Nothing above the `## Scoring log` line is
edited afterwards.

## Why this arm exists

Section 5 of the manuscript does three things in five sentences, and the third contradicts the
first two.

1. It quotes the FLOP proxy: selection at `n=64` costs `61.3x` the metered decoder's forward
   passes.
2. It then says that **measured** on one card the ratio is `35.4x`, and that the measurement
   *overturns where the cost sits*: the reward pass is `9.3%` of selection's wall clock and the
   draws are `90.7%`, so **"the lever is `n`, not the scorer"**
   (`results/serving_latency.csv`, caution (ae)).
3. It then concedes, in the most damaging sentence the paper contains: *"Held to the metered
   decoder's own compute the mechanism loses: at `n=4` and `0.92x` the cost it gains `-0.0395`
   `[-0.0720, -0.0065]` against it."*

**The `0.92x` in (3) is the proxy from (1), not the measurement from (2).** It is
`cost_ratio(n=4, 0.494B)` in `analysis/compute_matched.py`, which is `n (P_anchor + P_scorer) /
(P_anchor + P_risky)` -- parameter counts. That model prices the arm by *shrinking the scorer*,
which is exactly the lever (2) measured to be worth at most `9.3%`. So the paper selects its
compute-matched cell on an axis its own measurement rejects, and the appendix says so in the same
paragraph: *"a `0.5`B scorer divides the price by `4.16` at every `n`"*, which is a FLOP statement
the wall clock does not support.

Same class as cautions (ae), (ah) and (am): the number is computed correctly and is not the
quantity the sentence around it claims. It matters more than those because a reader who takes the
paper's own `35.4x` and `9.3%` and does the arithmetic can catch it, and because caution (ag) says
a concession is the worst place in a paper to carry a wrong number.

**This arm cannot be settled by arithmetic**, which is why it needs a card rather than a spreadsheet.
Whether `sel05b_n4` is or is not compute-matched depends on how the serving harness batches: draws
at `n` are `n` seed groups of up to `--batch-size` prompts each (`dap/e1.py:_run_seed_group`), so
`draws(n)` is linear in `n` at full batch width and NOT `n` times a single generation. The
extrapolation from the committed `n=64` measurement is a proxy for a proxy. Measure it.

## Where it runs, and why it cannot run here

A timing arm requires one exclusively-held card on an otherwise-quiet box. This host has none:
GPU 2 is running feat-134 and GPUs 0, 1 and 4 carry another project's job. **Host B is idle on all
eight H100s**, so the measurement goes there, on ONE card, with the other seven left idle for the
duration -- a wall-clock number measured beside seven busy cards is not a wall-clock number.

**The committed `35.4x` is a local A100 measurement and this arm does not replace it.** Host B is
different silicon; its ratios are internally consistent and are reported as a second measurement
beside the first, never as a correction to it. The `n=64` cell is measured on both, which is what
lets a reader translate between them.

## What runs

One H100, box otherwise idle, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `HF_HUB_OFFLINE=1`,
`scripts/run_cost_grid.sh`. Every cell serves the **same 40 neutral prompts**, `--max-new-tokens
200`, `--batch-size 64`, so each cell delivers the same work and a ratio of seconds is a ratio of
serving cost.

| cell | command shape |
|---|---|
| metered `k=10`, one and two completions | `h1.py --k-values 10.0 --trajectories-per-prompt 1` and `2` |
| draws at `n` | `h1.py --k-values 0.0 --trajectories-per-prompt n`, `n` in `1, 2, 4, 8, 16, 64` |
| reward pass at `n` | the first `n` candidates per prompt, `Qwen2.5-7B-Instruct` and `Qwen2.5-0.5B-Instruct` |

Interleaved `MET1, MET2, n=64, n=16, n=8, n=4, n=2, n=1` and then the same list reversed, so every
cell is measured twice and box drift cancels between reps (caution (ae): the metered arm's spread
was `15.3%` on a 25-second job, so one rep is not a measurement). Both reps are reported, never
only their mean.

**Loading a model is not serving, and it is inside every timed cell.** `h1.py` loads its
checkpoints on every invocation, so a cell's seconds are `load + n x work`. That overhead is a few
seconds against a few seconds at `n=1` and it is exactly the region where the matched cell will
be, so leaving it in would put the answer where the loader is rather than where the work is -- and
it would do so asymmetrically, since the metered path loads an `8`B model the selection path never
touches. A server loads once and then serves, so the quantity that decides this arm is the
**marginal** one:

- fit `draws(n) = a + b n` by least squares over the six `n` and both reps; `b` is the cost of one
  more completion per request and `a` is the loader.
- the metered path is measured at one and two completions, so `b_met = met(2) - met(1)` and
  `a_met = met(1) - b_met` directly, with no fit.
- the reward pass is timed inside our own code, which reports its load and its scoring separately,
  so `reward(n, scorer)` is scoring alone.
- `cost(n, scorer) = n b + reward(n, scorer)` and `ratio(n, scorer) = cost / b_met`.

The raw ratios, loader included, are reported in the same table. They are the number a deployer
sees if they restart the server for every request, and they are not what B1 reads.
Judging is never timed: a deployer serves, it does not judge.

## Gates, read before any band

- **G0 (instrument).** `ratio(64, 7.6B)` must land in `[15, 70]`. Deliberately wide: this is
  different silicon from the `35.4x` on record, and caution (as) is the standing lesson that a
  gate calibrated on an assumed floor rather than a measured one fails everything including the
  thing it was validating. What `[15, 70]` excludes is a broken pipeline -- a ratio near `1`
  (the arms are not what they claim) or near `500` (a card shared with something). If G0 fails,
  **nothing below is read or quoted.**
- **G1 (the premise).** The reward share `reward(64, 7.6B) / cost(64, 7.6B)` must be `< 0.25`.
  The committed local value is `0.093`. This is the finding the entire repair rests on; if the
  scorer turns out to be the majority of the clock on this hardware, the proxy was right about the
  shape and **the repair is withdrawn, not adjusted.**
- **G2 (same work served).** Mean served tokens per request must agree between the metered cell
  and the `n=64` cell within `5%`. If they do not, the seconds are still reported but every ratio
  is additionally normalised per served token and both are shown.
- **G3 (the linear model is allowed).** The fit `draws(n) = a + b n` must reach `R^2 >= 0.98` and
  `b > 0`. Below that the marginal decomposition is rejected outright: B1 is then read off the raw
  ratios, and the arm says so rather than quietly using a fit that does not describe the data.

## Bands, committed before the data

- **B1 -- the matched cell.** Over the measured grid, the compute-matched cell is
  `argmin |ratio - 1|` among cells with `ratio` in `[0.70, 1.45]`, on the marginal ratio if G3
  passes and the raw ratio if it does not. If no cell falls in that window, the reading is
  **NO MATCHED CELL** and the two cells bracketing `1.0` are reported instead. The rule is fixed
  here so the cell cannot be chosen after its judged gain is seen.
- **B2 -- is the paper's cell compute-matched?** Measure `ratio(4, 0.494B)` against the `0.92x`
  the manuscript prints.
  - **CONFIRMED** if the measured ratio is in `[0.70, 1.45]`: the sentence stands, with a measured
    ratio replacing the proxy.
  - **MISPRICED** otherwise, and the sentence is rewritten at whatever B1 selects.
- **B3 -- the fate of the concession, all three branches fixed in advance.** Let `g*` be the judged
  gain of the B1 cell and `g_met = +0.0400` the metered decoder's, both already committed in
  `results/compute_matched.csv`; the paired difference is recomputed by
  `analysis/compute_matched.py` under its own F5 replication gate and by nothing else.
  - **CONCESSION STANDS** if the paired difference is negative and its interval excludes zero:
    the sentence is restated at the measured cell, with the measured ratio.
  - **CONCESSION WITHDRAWN** if the paired difference is positive and its interval excludes zero:
    the paper says so, in the body, and names the proxy as the reason it had the opposite
    sentence. This branch is written here precisely so that finding it cannot look like a rescue.
  - **UNRESOLVED** if the interval contains zero: the sentence becomes that at matched measured
    compute the two are indistinguishable, and the interval is printed.
  - If F5 fails, B3 reads **NOT SCORED** and the manuscript keeps the `n=4` judged number while
    naming its measured cost, because a cost correction does not need a judged re-run to be true.

## What we predict, so that being wrong is visible

Draws are `n` full-width seed groups, so `draws(n)` should be close to linear in `n` and
`b` should be about half `b_met`, since the metered path runs two models and a KL solve at every
step where selection runs one. We therefore predict **B1 selects `n=2`**, **B2 reads
MISPRICED with `ratio(4)` near `2.0`** -- a factor of about `2.2` above the printed `0.92x` -- and
**B3 reads CONCESSION STANDS**, with the loss at the matched cell no smaller than the one on
record. That is the direction that does not flatter the paper: it says selection at honestly
matched compute buys even less than the paper currently concedes.

## Excluded in advance

- Replacing the committed `35.4x` with a host-B number, or quoting a host-B second beside a local
  second. Ratios are compared; seconds are not.
- Regenerating, re-scoring or re-judging anything to make a cell match. The judged gains are the
  ones already in `results/compute_matched.csv`; the only new data this arm produces is wall clock.
- Choosing the matched cell by anything but `argmin |ratio - 1|`, or widening `[0.70, 1.45]` after
  seeing where the cells fall.
- Quoting any band if G0 or G1 fails.
- Timing the judge into `cost`. A deployer does not judge, and including it would flatter the
  metered decoder.
- Reading B1 off the raw ratios when G3 passes, or off the marginal ones when it fails. The choice
  is made here, by the gate, and not after the two tables are side by side.
- Treating a `0.5`B reward time as zero. It is measured, not assumed.

## Compute

One H100, about `90` minutes: `95` seed-group generations per rep over two reps, plus six metered
generations and twenty-four reward passes. Under the 24-GPU-hour escalation rule by two orders of magnitude.

## Scoring log

### Note written 2026-09-21 17:58, while the run was still generating and before any reward cell existed

Seven cells are on disk --- the two metered ones and draws at `n = 64, 16, 8, 4` --- and no reward
pass has run yet, so no `ratio` has been computed by anything. From those seven the slope is
already visible (`454.850`, `122.097`, `66.067`, `38.297` seconds, and `18.002` / `24.668` for the
meter), which means `64 b / b_met` alone is about `66`, and G0's upper end is `70`. **G0 is
therefore likely to fail, and the reason is a defect in this registration rather than in the
measurement.** It is recorded now, with the number that would decide it still unmeasured, because
recording it afterwards is worth nothing.

**The defect.** G0's endpoints were anchored on the committed `35.4x`, which is a **loader
inclusive** ratio --- `908.031 / 25.637` out of `results/serving_latency.csv`, both arms carrying
their model load. G0 was then written against `ratio`, which this same registration defines two
sections earlier as the **loader excluded** one, `cost / b_met`. A band taken from one quantity
and applied to a different one is caution (as)'s exact error, committed here in our own
pre-registration, and it is worse than caution (as)'s because that band was merely too tight
whereas this one is about the wrong measurement.

**What will be done if it fires.** Not a quiet widening, and not a switch of G0 to the raw ratio
after seeing which of the two passes. The arm will be reported as **INVALID BY OUR OWN
SPECIFICATION** (caution (w): a defect in our specification must not retire a question), and G0
re-registered as the band the paper's own committed prices imply for this cell: it prints `35.4x`
(measured, raw) and `61.3x` (the FLOP proxy) for `n=64`, and any measurement of that cell within a
factor of two of either is not a broken pipeline, so the re-registered band is
`[0.5 x 35.4, 2 x 61.3] = [17.7, 122.6]`. That is derived from two numbers committed before this
arm existed and from the sentence in G0 that already says what the gate is for -- "a ratio near 1
... or near 500" -- and not from today's measurement. The re-scoring uses the same data; nothing
is re-run, and this paragraph is what makes the ordering auditable.

**Why the marginal ratio is nonetheless the right quantity, and is not being chosen to suit.** A
served request does not pay for loading a checkpoint; a server loads once and serves for hours. The
raw ratio flatters the metered decoder specifically, because loading is `11.3` of its `18.0`
seconds and only `10.5` of selection's `454.9`. Both ratios are reported in `cost_grid.csv` either
way, which is the reason that table was specified to carry both.

## Scoring, 2026-09-21

Run on host B, GPU 0, the other seven cards idle throughout (`box at start` and `box at end` both
read `1 MiB` on every other card, and the only compute process on the box was ours). Forty cells,
every one twice.

```
bash scripts/run_cost_grid.sh 0 40
.venv/bin/python analysis/cost_grid.py --report --out results
```

Outputs `results/cost_grid.csv` and `results/cost_grid_bands.csv`.

### The repeats, first, because a timing arm with one rep is not a measurement

| cell | rep 1 | rep 2 | spread |
|---|---|---|---|
| metered, one completion | `18.002` | `17.887` | `0.6%` |
| metered, two completions | `24.668` | `25.167` | `2.0%` |
| draws `n=64` | `454.850` | `461.183` | `1.4%` |
| draws `n=16` | `122.097` | `119.920` | `1.8%` |
| draws `n=1` | `17.951` | `17.895` | `0.3%` |

The committed local arm reported a `15.3%` spread on the metered cell and called it out as the
reason one rep is not a measurement. Here it is `0.6%`, and the difference is the box: that arm ran
on a card that was exclusively ours inside a machine where two other cards held `25` and `68` GB of
someone else's job, and this one ran on a machine with nothing else on it.

### The fit, which is the whole reason this arm needed a card

`draws(n) = 10.31 + 6.99 n` seconds, **`R^2 = 0.99990`** over twelve points. The metered path,
measured directly at one and two completions rather than fitted, gives `b_met = 6.97` s and
`a_met = 10.97` s.

**`b = 6.99` against `b_met = 6.97`: one draw from the anchor costs what one metered decode costs,
to `0.3%`.** That is the measurement the rest of this follows from, and it was not what we
predicted --- we predicted about half, since the metered path runs a `1.76`B anchor and an `8.03`B
risky model and a KL solve at every step where selection runs the anchor alone. At batch `40` and
`204` steps neither path is weight-bound; both are dominated by per-step overhead, so carrying a
model `4.6x` larger through the loop costs essentially nothing extra and drawing a second
candidate costs a whole second decode.

### Gates

| gate | value | reading |
|---|---|---|
| G0 instrument, `ratio(64, 7.6B)` in `[15, 70]` | `67.243` | **PASS** |
| G1 premise, reward share `< 0.25` | `0.0458` | **PASS** |
| G2 same work served, within `5%` | `0.0000` | **PASS** |
| G3 linear model allowed, `R^2 >= 0.98`, `b > 0` | `0.99990` | **PASS** |

**G0 passed by `2.757` and the note above it, written while the reward cells were still unrun,
predicted it would fail.** It did not, and the band's defect is real anyway: its endpoints came
from a loader-inclusive reference and it was applied to a loader-excluded quantity, which is why it
landed `2.8` short of excluding a perfectly sound measurement. Recording a defect and then not
needing the repair is the outcome that costs nothing; the note stays where it is, and the
re-registration it describes was not used.

G2 reads exactly `0.0000`: both paths served `204.5` tokens per request on the same `40` prompts.

### B1 -- the matched cell is `n=1`, and we predicted `n=2`

| scorer | `n` | marginal | raw | FLOP proxy |
|---|---|---|---|---|
| `0.494`B | `1` | **`1.074x`** | `1.111x` | `0.240x` |
| `0.494`B | `2` | `2.083x` | `1.490x` | `0.481x` |
| `0.494`B | `4` | `4.109x` | `2.277x` | `0.961x` |
| `7.6156`B | `64` | `67.243x` | `26.933x` | `61.289x` |

`argmin |ratio - 1|` inside `[0.70, 1.45]` selects **the `0.494`B scorer at `n=1`, `1.074x`**. It is
the only cell in the window: `n=2` is already `2.083x`, because `ratio(n) ~ n`.

### B2 -- MISPRICED, and by more than four times

`sel05b_n4` measures **`4.109x`** the metered decoder against the **`0.92x`** Section 5 prints, a
factor of **`4.47`**. We predicted `2.0x` and a factor of `2.2`, and were wrong in the same
direction and twice as far, for the same reason B1 moved: the proxy divides selection's price by
`4.16` when the scorer shrinks from `7.6`B to `0.494`B, and on the clock that swap is worth `2.4%`
of the cell's cost (`1.735` s against `0.687` s inside `29.7` s).

### B3 -- CONCESSION STANDS

The matched cell is a single draw, which is the anchor serving its own sample: its gain is `0` by
construction, against the metered decoder's `+0.0400`. **At the metered decoder's own measured
compute, selection is not a mechanism at all.** The concession in Section 5 survives; what changes
is that it was understated. The paper says selection loses by `-0.0395` at a cell it prices at
`0.92x`; that cell actually costs `4.1x`, and at `1.07x` selection has no draws to choose from.

### A deviation from this registration, declared

B3 was registered as a re-run of `analysis/compute_matched.py` gated by its F5 replication check.
It was computed instead from `results/compute_matched_per_prompt.csv`, which that same script
already committed --- the per-prompt judged gain of every cell --- so **no second judging pass was
run and there is no cross-pass drift to gate** (caution (ap)). What replaced F5 is stronger, not
weaker: `paired_boot` draws from one `Random(9163)` stream that the script advances once per arm
and then once per band, so reproducing a committed band requires replaying that entire sequence,
and the replay lands on F4's committed `-0.0395 [-0.0720, -0.0065]` **exactly, at both interval
ends**. `tests/test_cost_grid.py` fails if it ever stops doing so. The deviation is recorded
because it is a deviation, not because it is a weakness.

### Two findings this arm was not registered to make

**First: the manuscript's explanation of its own serving measurement is falsified.**
Appendix~I says *"the FLOP model is `1.73x` pessimistic as a price, because `64` short completions
generated in one batch do not cost `64` sequential ones"*. They do. `draws(n)` is linear in `n` at
`R^2 = 0.99990`, and the code says why: `dap/e1.py:_run_seed_group` batches across **prompts**
within one seed, and `--trajectories-per-prompt n` is realised as `n` separate seed groups, so
drawing `64` candidates is `64` sequential batched decodes with no amortisation across the
candidate dimension whatsoever. The same code produced the local arm. Whether a server that
batched the `n` candidates together could recover the amortisation is untested and this arm says
nothing about it --- but neither does the `35.4x`, and the sentence claims it does.

**Second: the gap between `61.3x` and `35.4x` is the model loader, and it sits in the
denominator.** Of the metered path's `17.94` seconds, `10.97` are loading checkpoints and `6.97`
are serving; of selection's `458.0` at `n=64`, `10.31` are loading. A ratio of totals therefore
divides by a number that is `61%` startup, and it flatters the metered decoder specifically.
On host B the two conventions read `26.9x` (loader included) against `67.2x` (loader excluded), a
factor of `2.5`, and the loader-excluded figure is `1.10x` **above** the FLOP proxy rather than
`1.73x` below it. A server loads once and serves for hours, so the loader-excluded ratio is the one
a deployer pays.

**The committed `35.4x` is not revised by this and must not be.** It is a local A100 measurement,
this is an H100, and the registration excluded the substitution in advance. What the paper gains is
the convention: `35.4x` is a ratio of totals with both loaders inside it, and the second host shows
what that convention costs.

### What goes into the manuscript

1. Section 5's compute-matched clause is restated at the measured cell, and the `0.92x` goes.
2. Appendix~I's batching explanation is withdrawn and replaced by the loader, measured.
3. Every site quoting `35.4x` gains the word that makes it a ratio of totals. The number stays.
