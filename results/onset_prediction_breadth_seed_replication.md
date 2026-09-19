# Pre-registration: do the breadth ladder's verdicts survive a fresh draw?

**feat-138.** Written and committed **before any of these arms runs**. Nothing above the
`## Scoring log` heading is edited afterwards.

## Why this arm exists, and why it is not optional

`feat-136` scored four anchors on the second host. Three returned **SATURATED BY 8** at
$0.19$--$0.38$ interval half-widths and its own committed rule prints, for each of them:

> MARGINAL: not promoted into the manuscript without a replication, whatever the interval says.

So the three nulls that carry hypothesis **H2** --- *no non-Comma anchor clears $2.0$ half-widths* ---
**cannot be used until they are re-drawn.** This arm does that. The fourth, Comma-7B (1T), read
**CLIMBS** at $2.34$ half-widths, above the boundary, where caution (ap)'s rule predicts a reading
survives; it is included so the rule is tested out of sample a third time rather than only relied on.

## What changes, and what is held

**Exactly one thing changes: `--seeds 42 43 44` $\to$ `52 53 54`.**
`dap/stats.py:build_trajectory_seeds` hashes `(prompt_id, seeds, n)` into the high 16 bits, so the
$64$ trajectory seeds are disjoint from the original $64$ by construction.

`--batch-size 32` **is held**, and this is the whole lesson of `feat-132`: that arm changed the seed
AND the batch size, declared the second in advance as "a second, independent re-roll of the *same*
distribution", and its own integrity gate falsified the second clause --- the draw-0 empty fraction
moved $0.094 \to 0.056$, all of it in the `neutral` class, two-proportion $z = 2.78$. An arm that
changes two things is not the comparison it was registered to be and is **INVALID** rather than a
failed replication (cautions (u), (v), (w)). `scripts/run_breadth64_seed.sh` differs from
`scripts/run_breadth64.sh` by exactly one flag, and that is checked mechanically, not by reading.

**Both draws are on the SAME HOST through the SAME pipeline.** These are `role=new` arms and no
host-transfer gate applies to them: the comparison is host B seed-42 against host B seed-52. That is
deliberate --- it isolates the draw, which is the only thing this arm asks about.

## The arms and their committed bands

Each band is the reading `feat-136` recorded on `2026-09-19`, derived by
`analysis/score_breadth_ladders.py` from the committed per-prompt CSVs:

| arm | first draw, $g(64)-g(8)$ | half-widths | verdict | predicted verdict on re-draw |
|---|---|---|---|---|
| `comma1t` | $+0.0970\ [+0.0560, +0.1390]$ | $2.34$ | CLIMBS | **CLIMBS** (above the $2.0$ boundary) |
| `pleias350m` | $+0.0090\ [-0.0320, +0.0490]$ | $0.22$ | SATURATED BY 8 | **SATURATED BY 8** |
| `kl3m170m` | $+0.0130\ [-0.0210, +0.0470]$ | $0.38$ | SATURATED BY 8 | **SATURATED BY 8** |
| `kl3m520m` | $-0.0070\ [-0.0430, +0.0300]$ | $0.19$ | SATURATED BY 8 | **SATURATED BY 8** |

**G1 --- verdict agreement.** Each arm's verdict is recomputed by the same rule (`lo > 0` CLIMBS,
`hi < 0` TURNS OVER, else SATURATED BY 8) and must equal the prediction above. A disagreement is
**the finding**, not a nuisance, and is reported as such.

**G2 --- distance.** $|D_{s52} - D_{s42}| \le 0.0610$ for every arm. That number is
`MAX_SEED_MOVE` in `analysis/score_breadth_ladders.py`: the largest seed-replication move ever
measured here (KL3M-1.7B, at $1.73$ half-widths). It is a constant that **predates this arm** and is
not chosen from anything it has to judge. The prediction is the RANGE only --- caution (ap) says the
half-width ratio predicts the **verdict** and never the **distance**, citing $1.71 \to 0.061$,
$2.12 \to 0.000$, $2.43 \to 0.013$, which do not order.

**G2 does not override G1.** A verdict that agrees while the distance exceeds $0.0610$ is reported as
a verdict that agreed and a distance that did not, in those words.

## Integrity checks, fixed before the data exists

**I1 (BLOCKS) --- mean words GIVEN NON-EMPTY within $5\%$ of the first draw.** Not the raw mean.
This is the repair of the statistic `feat-136`'s G0b used, registered here for a new arm rather than
applied to an old one: on `2026-09-19` `comma7bhb` failed a $5\%$ tolerance on the raw mean at
$+5.3\%$, and the decomposition showed the mean is exactly
`(1 - empty_frac) x (mean words given non-empty)` --- $(1-0.094)\times109.47 = 99.18$ locally and
$(1-0.020)\times106.54 = 104.41$ on host B --- so the **whole** failure was the empty rate while
length given non-empty moved $-2.7\%$. Caution (v) already recorded that an aggregate gate on a
stratified rate gates the wrong quantity. **Disclosed plainly: that decomposition was known before
this check was written.** It is registered here, before this arm's data exists, and it is NOT applied
retroactively to `comma7bhb`, whose band stays uncomputed and unread.

**I2 (REPORTED, blocks only at $|z| > 3$) --- the empty fraction**, by two-proportion $z$ against the
first draw, per class and on the total. With the host, the pipeline and the batch size all held, only
the seed moves it, so a large shift would mean something other than the draw changed. An absolute
tolerance is not used: caution (v) records that $0.03$ is unfalsifiable at KL3M ($0.000$--$0.002$)
and tighter than the arm-to-arm spread at Comma-7B ($0.030$--$0.094$).

**I3 (BLOCKS) --- coverage:** the grid $\{1,2,4,8,16,32,64\}$ on all $500$ prompts, judge B present.

## What this arm cannot do

It re-draws the trajectory pool on one host. It says nothing about hardware transfer, which is a
separate question `feat-136` asks and which its two host arms currently do not answer. It cannot
turn a MARGINAL reading into a stable one --- if a null replicates, the null is a null that
replicated at $0.2$ half-widths, and the honest description of a $0.2$-half-width reading is that its
interval contains zero comfortably, not that the effect is known to be absent. It adds no new anchor
and so cannot address the scale confound in H2, which is that the only clean public-domain-trained
$7$B models are the two Comma checkpoints.

## Compute

Four arms, $500 \times 64$ generations each at $\le 7$B with `--max-new 200`, on four idle H100s.
`compute_hours.csv` records `sel_comma7b_64` at $13.42$ A100-hours for the same workload at $7.6$B;
the three small anchors finished in $2.5$--$3$ wall-clock hours each on this host on `2026-09-19`.
Estimate **$\approx 11$ gpu-hours total, no single arm above $4$** --- under the $24$-gpu-hour
escalation threshold. Run under the user's instruction of 2026-09-20 to use host B's GPUs maximally.

## Scoring log

### 2026-09-20 --- all four verdicts survive the re-draw, and every move is small

| arm | first draw | re-draw (seeds 52 53 54) | half-widths | predicted | read | move |
|---|---|---|---|---|---|---|
| Comma-7B (1T) | $+0.0970$ | $+0.0860\ [+0.0440, +0.1270]$ | $2.07$ | CLIMBS | **CLIMBS** | $0.0110$ |
| Pleias-350M | $+0.0090$ | $+0.0020\ [-0.0380, +0.0400]$ | $0.05$ | SATURATED | **SATURATED** | $0.0070$ |
| KL3M-170M | $+0.0130$ | $+0.0210\ [-0.0150, +0.0570]$ | $0.58$ | SATURATED | **SATURATED** | $0.0080$ |
| KL3M-520M | $-0.0070$ | $+0.0110\ [-0.0240, +0.0480]$ | $0.31$ | SATURATED | **SATURATED** | $0.0180$ |

**G1: four of four verdicts agree. G2: every move is inside $0.0610$**, and in fact inside
$0.0180$ --- a quarter of the bound. I1, I2 and I3 pass on every arm; the largest length move given
non-empty is $+3.2\%$ and the largest empty-fraction $|z|$ is $1.24$.

**The three MARGINAL nulls carrying H2 are now replicated**, which is what `feat-136`'s own rule
demanded before they could be used. And caution (ap)'s boundary rule held again: the one arm above
$2.0$ half-widths ($2.34$) reproduced at $2.07$ and moved $0.0110$, while the three below stayed
below and stayed nulls.

**What this establishes, and it is the control the next entry needs:** on ONE host, through ONE
pipeline, with only the trajectory pool re-drawn, this measurement moves by $0.007$ to $0.018$ and
**never changes a verdict**. That is the size of a pure re-draw here.
