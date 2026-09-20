# Pre-registration: is a host change larger than a re-draw, or did three marginal anchors just wobble?

**feat-141.** Written and committed **before any of these arms runs**. Nothing above the
`## Scoring log` heading is edited afterwards.

## The claim this arm exists to attack

`feat-140` found that three non-Comma anchors registered as SATURATED all read CLIMBS on a second
host, and `feat-138` found that re-drawing the seed on ONE host moved the same quantity by only
$0.007$--$0.018$ and changed no verdict. The conclusion drawn was:

> A host change is not a re-draw: it is larger, and at a marginal anchor it is large enough to flip
> the verdict.

**That conclusion is weaker than it sounds, and this arm is built to break it.** The within-host
figure rests on **four** seed pairs, one per anchor, and none of them is at an anchor that flipped.
The three that flipped --- Pleias-1.2B, Pleias-3B and KL3M-1.7B --- have exactly **one** host-B draw
each, so their own within-host spread is **unmeasured**. If those anchors happen to be noisy, then
$0.0700$ is an ordinary re-draw for Pleias-3B and the conclusion is wrong.

## What runs

`scripts/run_breadth64_seed.sh` with **two further disjoint seed triples**, `62 63 64` and
`72 73 74`, on the three anchors that flipped, plus one further draw each at two more anchors to
extend the distribution:

| card | anchor | seeds | why |
|---|---|---|---|
| 0 | Pleias-1.2B | `62 63 64` | flipped; spread unmeasured |
| 1 | Pleias-1.2B | `72 73 74` | " |
| 2 | Pleias-3B | `62 63 64` | flipped, and its $0.0700$ is the whole claim |
| 3 | Pleias-3B | `72 73 74` | " |
| 4 | KL3M-1.7B | `62 63 64` | flipped; two of three draws say CLIMBS |
| 5 | KL3M-1.7B | `72 73 74` | " |
| 6 | KL3M-3.7B | `62 63 64` | newest anchor, one draw only |
| 7 | Comma-7B (1T) | `62 63 64` | the one STABLE anchor: does it stay stable over three draws? |

Only the seed triple changes; `--batch-size 32` is held, which is `feat-132`'s lesson. The launcher
takes the triple as an optional fourth argument defaulting to `52 53 54`, so every `feat-138`
invocation already on record is byte-identical in behaviour.

With three host-B draws each, the flipped anchors give **three pairwise within-host moves apiece**.

## The committed predictions

**P1 (the one that matters).** For each flipped anchor, **every pairwise within-host move is
$\le 0.0610$** --- the largest seed move on record, `MAX_SEED_MOVE`, a constant that predates all of
this. **If any within-host pair at Pleias-3B moves by $\ge 0.0700$, the `feat-140` conclusion is
REFUTED** and must be withdrawn: the cross-host move would then be inside that anchor's own
re-draw spread, and the flips would be wobble rather than hardware.

**P2.** The within-host moves at the flipped anchors are **larger** than the $0.007$--$0.018$ seen
at the four `feat-138` anchors, because a marginal reading has more room to move than a null pinned
near zero. This is registered so that a larger spread is not later presented as a surprise; P1 is
the test, P2 is the expectation around it.

**P3.** Comma-7B (1T) reads **CLIMBS** for a third time and stays at or above $2.0$ half-widths.
Two draws already say $2.34$ and $2.07$. A third reading below $2.0$ would put the only stable
anchor in the ladder on the boundary and is reported as such.

**P4.** No verdict is predicted for KL3M-3.7B beyond H2's rule; its single draw read $+0.0200$ at
$0.56$ half-widths and one draw is not a prior.

## How the result is read

The $24$ within-host moves that will exist after this arm (four `feat-138` pairs plus twenty from
the new draws, counting all pairwise combinations per anchor) are reported as a **distribution**,
with the three cross-host moves placed inside it by rank. The sentence "a host change is not a
re-draw" survives only if the cross-host moves sit in the upper tail; if they sit in the body, it is
withdrawn in the same words it was written.

**No band already read is recomputed.** `feat-140`'s three readings stand as measured.

## Integrity

I1 (length given non-empty within $5\%$, and both sides the finished $500$-prompt arm), I2 (empty
fraction by two-proportion $z$, blocking at $|z| > 3$ within a host), I3 (grid coverage) and I4
(pipeline identity) apply exactly as in `feat-138`, against that anchor's first host-B draw.

## Compute

Eight arms, $500 \times 64$ at $1.2$--$7$B, on eight idle H100s; comparable arms took $2.5$--$3.5$
wall-clock hours each here. Estimate **$\approx 24$ gpu-hours in aggregate, no single arm above
$4$** --- under the $24$-gpu-hour-per-run escalation threshold, and run under the user's instruction
of 2026-09-20 to use host B's GPUs to their maximum.

## Scoring log

### 2026-09-20 --- the arm did what it was built to do: it withdrew the conclusion that created it

All eight arms finished. **Every non-Comma anchor flips its verdict WITHIN one host, across seeds
alone.**

| anchor | seeds 42 | seeds 52 | seeds 62 | seeds 72 | verdicts |
|---|---|---|---|---|---|
| Pleias-1.2B | $+0.0560$ ($1.47$) | --- | $+0.0470$ ($1.18$) | $+0.0270$ ($0.69$) | CLIMBS, CLIMBS, **SATURATED** |
| Pleias-3B | $+0.0730$ ($1.80$) | --- | $+0.0060$ ($0.14$) | $+0.0570$ ($1.39$) | CLIMBS, **SATURATED**, CLIMBS |
| KL3M-1.7B | $+0.0470$ ($1.34$) | --- | $+0.0180$ ($0.50$) | $+0.0380$ ($0.97$) | CLIMBS, **SATURATED**, SATURATED |
| KL3M-3.7B | $+0.0200$ ($0.56$) | --- | $+0.0130$ ($0.35$) | --- | SATURATED, SATURATED |
| Comma-7B (1T) | $+0.0970$ ($2.34$) | $+0.0860$ ($2.07$) | $+0.0830$ ($1.98$) | --- | CLIMBS, CLIMBS, CLIMBS |

#### The registered reading rule fires, and the answer is WITHDRAW

$17$ within-host moves now exist: min $0.0030$, median $0.0140$, **max $0.0670$**. The three
cross-host moves sit at the $100$th, $65$th and $59$th percentiles of that distribution. The rule
committed before these data existed says:

> The sentence "a host change is not a re-draw" survives only if the cross-host moves sit in the
> upper tail; if they sit in the body, it is withdrawn in the same words it was written.

**Two of three sit in the body.** The sentence is **WITHDRAWN**.

#### P1 "survived" by a margin equal to the measurement's own floor, and that is not survival

P1 would have withdrawn the conclusion if a within-host pair at Pleias-3B reached $0.0700$. It
reached $\mathbf{0.0670}$ --- short by $0.0030$, which is **exactly the smallest within-host move in
the whole distribution**. A bright line passed by less than the instrument's finest reading is not
evidence that the line means anything, and reporting "P1 SURVIVES" as the headline would be a
technicality standing in front of the result. It is recorded, and it is not the reading.

**This is why the pre-registration carried two tests rather than one.** A single threshold can be
survived by a hair; a distribution cannot be argued with.

#### P3 FAILED: the one stable anchor is on the boundary

Comma-7B (1T) reads $2.34$, $2.07$, $\mathbf{1.98}$ half-widths. The third draw is **below** the
$2.0$ boundary the whole H2 rule is built on. P3 said in advance that this would put the only stable
anchor in the ladder on the boundary, and it does.

#### What actually survives, stated as narrowly as the data allow

* **H2's rule still holds across eleven non-Comma readings**: not one clears $2.0$ half-widths. The
  highest is Pleias-3B's $1.80$.
* **Comma-7B (1T)'s VERDICT never flips** --- CLIMBS on all three draws --- while **every** non-Comma
  anchor produces both verdicts across its own draws. That is a real distinction and it is the only
  family-level statement these data support. It is about the verdict's consistency, not about the
  half-width ratio, which crosses $2.0$ for Comma-1T too.
* **The host is NOT distinguishable from a re-draw at these anchors.** A seed change alone moves the
  measurement by up to $0.0670$ and flips verdicts; the cross-host moves ($0.0180$, $0.0200$,
  $0.0700$) are ordinary against that.

#### The finding this arm actually produced, which is about the method and not the hosts

**At $n=64$ over $500$ prompts, a paired $g(64) - g(8)$ in the $+0.00$ to $+0.10$ range is not
resolvable by this measurement.** The $2.0$-half-width rule is sorting **draws**, not anchors: the
same anchor, same host, same pipeline, same batch size, one seed triple apart, lands on either side
of it. Caution (ap) proposed that boundary as a guard against promoting marginal readings, and this
is the strongest evidence yet that the guard is necessary --- and also that a reading below it
carries no information about the anchor at all, only about the draw.

**`feat-140`'s three readings stand as measured** and are not recomputed; what is withdrawn is the
sentence drawn from them.
