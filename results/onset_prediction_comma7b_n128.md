# Pre-registration: where is Comma-7B's ceiling?

**feat-134.** Written and committed **before the arm runs**. Nothing above the `## Scoring log`
heading is edited afterwards.

## Why

The paper says, in `sections/appendix_selection.tex`, *"We did not take Comma-7B past $64$, so where
the strongest anchor's ceiling sits is open."* This closes that sentence.

What is already known. At the **audited** anchor the ceiling is bracketed: `feat-129` took TinyComma
to $n=128$ and read paired $g(128)-g(64) = +0.0140$ $[-0.0180, +0.0460]$, an interval containing
zero, so \textsc{saturated by 64}. At **Comma-7B** the curve is still rising where its grid stops ---
$+0.041, +0.060, +0.072, +0.103, +0.127, +0.173$ at $n = 2 \ldots 64$ under judge~B, Spearman
$+1.000$ --- and its paired $g(64)-g(8) = +0.1010$ $[+0.0590, +0.1420]$ **replicated** under a
disjoint seed draw (`feat-133`, $+0.0880$ $[+0.0460, +0.1290]$). So this anchor climbs to $64$
reproducibly and nobody has looked past it.

This matters beyond one cell. The shape the paper claims is *the gain rises in $\log n$ to an
anchor-dependent ceiling and then stops*, and that claim currently rests on **one** measured ceiling.
A second one either confirms the shape at the strongest anchor or shows the two anchors disagree ---
and the deployment sentence ("which $n$ is worth paying for is a property of the anchor a deployer
must measure") is worth more with two brackets than with one.

## What is measured

$n=128$ on the same $500$ ordinary prompts, `--seeds 42 43 44`, `--trajectories-per-prompt 128`,
`--max-new-tokens 200`, the same reward and the same two judges as
`output/phase5/sel_comma7b_64`.

**`--batch-size` is deliberately not passed**, so `h1.py`'s default of `8` applies --- which is what
the arm on record used, its command block passing no such flag. Batch size is part of the seed
(caution `(u)`), and matching it is what makes the reproduction gate below possible at all.
`feat-130` asserted a batch size it had not checked and two of its three anchors' gates were
*inapplicable* rather than failed; this file states the premise and lets the gate test it.

## The first 64 of the 128 draws ARE the committed 64, and here is why

Not an assumption --- it follows from the code, and `feat-129` already measured it once
(`32{,}000` of `32{,}000` rewards bit-identical at ranks $0$--$63$ at TinyComma).

1. `dap/stats.py:build_trajectory_seeds` returns `(hash(base_seeds) << 16) | j` for draw $j$ and
   **does not depend on `prompt_id`**. So the first $64$ of a $128$-draw call are exactly the $64$.
2. `dap/e1.py:run_split_batched` groups jobs by that derived seed. Seed group $j$ therefore holds
   **one job per prompt --- the same $500$ jobs whether $n$ is $64$ or $128$.** Within a group the
   length bucketing is a deterministic sort and the batch slicing is deterministic, so the batches
   at ranks $0$--$63$ are identical between the two runs at the same batch size.
3. `a_patch/factory.py` calls `set_seed(batch_seed)` at the top of **every** `generate()`, so the
   number of seed groups processed before a given one cannot leak in through RNG state.

The one premise left is the batch size of the arm on record, and that is exactly what the gate tests.

## The reproduction gate --- on the reward cache, never on a judged number

Ranks $0$--$63$ of this arm's reward cache must be **bit-identical** to
`results/selection_rewards64_comma7b.csv` --- all `32{,}000` floats, compared with `==`.

**No $n>64$ number is read until it clears.** This is `feat-129`'s corrected gate, and the
correction matters: its first version compared judged `gain` across passes at `5e-4`, which cautions
`(e)` and `(m)` already forbade and which could never have passed. `32{,}000` floats compared
exactly is both sounder and far stricter than `28` summary cells compared loosely.

If the gate fails, the reading is **INAPPLICABLE, not a failure of the arm**: it means the arm on
record did not use batch size `8`, the two pools are different draws, and the correct response is to
chase the batch size --- not to read $n=128$ anyway. Excluded alternative 3 below forbids that escape.

## Committed band --- the paired $g(128)-g(64)$, judge~B, 500 prompts, within this pass

| reading | verdict |
|---|---|
| $>0$ with its interval excluding $0$ | **STILL CLIMBING.** The ceiling at the strongest anchor is **above** $128$, where the audited anchor's sits below it. The appendix then says the two measured ceilings differ and that the anchor-dependence claim is supported by anchors that disagree, which is stronger evidence for it than two that agree. |
| interval includes $0$ | **SATURATED BY 64.** The ceiling is bracketed between $64$ and $128$ at this anchor too, agreeing with the audited one. The appendix replaces *"where the strongest anchor's ceiling sits is open"* with the measured bracket. |
| $<0$ with its interval excluding $0$ | **TURNS OVER.** Reward overoptimisation is observed at this anchor on this grid, which Appendix~I currently says is *not* observed anywhere on its grid --- that sentence is then scoped to the arm it is about, and the turnover is reported as the finding it is. |

Whatever it reads, the sentence *"We did not take Comma-7B past $64$, so where the strongest
anchor's ceiling sits is open"* is replaced, and `tests/test_n128_frontier.py` --- which currently
asserts that sentence is present --- must be updated to assert whatever replaces it. That guard
firing is the point, not an obstacle (caution `(ag)`).

## Committed secondary, reported whatever it reads

The full eight-point grid under **both** judges; the reproduction gate's own count of matching
floats; and **the grid-dependence of the judged level**, which is this arm's free second measurement
of caution `(ap)`: the committed pass judged a **seven**-arm grid and this one judges **eight**, so
$g(64)$ is expected to move between them. At TinyComma that move was $+0.142 \rightarrow +0.076$.
Here it is reported beside the committed $+0.173$ as *information about the instrument* and is
**never** gated on, and no number from this pass is set against a number from the committed pass
except through the paired difference above, which is computed within one pass on both sides.

## Excluded in advance

We will not, after seeing results: change the scorer, either judge, the opponent, the prompt set,
the batch size or the $n$-grid; quote judge~A if judge~B disagrees; compare any judged *level* or
*gain* across the two passes as though it were a finding; **read any $n>64$ number if the
reproduction gate fails**; drop the $n \le 64$ rows and read $n=128$ anyway; re-run at other seeds
and pick the pass that agrees with the audited anchor; or put an $n=128$ number in the abstract,
which is about $n \le 64$ and $256$.

## What this arm cannot do, stated before it runs

It measures **one** anchor on **one** grid with **one** reward. If it reads \textsc{still climbing}
the ceiling is bracketed only from below, and $128$ is then a statement about where this grid
stopped rather than where the mechanism does --- the same honest limit `feat-129` stated at $n=64$
before going to $128$. It cannot separate a reward-model ceiling from an anchor-support ceiling;
Proposition~1 makes the anchor the ceiling in both readings, and the judge-free rows in
Appendix~I are where that separation is actually visible. It says nothing about extraction, which is
already measured to $n=256$ and reads $0.0000$.

## Compute --- and this is the arm that needed asking

Measured from the arm being extended, at the same batch size: `results/compute_hours.csv` has
`sel_comma7b_64` at **13.42 gpu-hours** for $32{,}000$ trajectories on one card. This arm is
$64{,}000$. Two cards are free --- **GPU 1 is another user's at $74$ GB and $100\%$, GPU 0 holds
their $597$ MiB, and GPU 3 is the 4 GB T400** --- so the split is uneven, because the class caps are
the only natural unit:

| card | classes | trajectories | expected |
|---|---|---|---|
| GPU 2 | neutral $200$ | $25{,}600$ | $\approx 11.2$ h |
| GPU 4 | creative $150$ + factual $150$ | $38{,}400$ | $\approx 19.0$ h |
| GPU 4 | merge + score to $n=128$ | --- | $\approx 0.8$ h |

**Total $\approx 31$ gpu-hours at $\approx 19.8$ hours of wall clock.** That is **over the
24-gpu-hour escalation threshold**, which is why this arm has sat unstarted in the handoff since
2026-09-17; it runs now because it was explicitly asked for. Rates are the ones `feat-133` measured
at this anchor and batch size, not estimates, and they already include the split penalty that arm
found (three cards bought $2.3\times$ the throughput, not $3\times$).

## Scoring log
