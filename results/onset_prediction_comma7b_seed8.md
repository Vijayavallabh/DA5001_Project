# Pre-registration: does Comma-7B's climb to $n=64$ survive an independent draw? (corrected protocol)

**feat-133.** Written and committed **before the arm runs**. Nothing above the `## Scoring log`
heading is edited afterwards.

## Why this exists, and what is different from `feat-132`

`feat-132` asked this same question and **could not answer it**. It changed two things where the
design intends one --- the seeds, and `--batch-size 8` $\rightarrow$ `32` --- having asserted in
advance that the second "does not change what is being drawn from." Its own integrity check
falsified that: the draw-0 empty fraction moved $0.094 \rightarrow 0.056$, concentrated entirely in
the `neutral` class ($45/200 \rightarrow 24/200$, two-proportion $z = 2.78$) where empties live at
this anchor, while `creative` and `factual` sit at $0.7\%$ and moved by $\pm 3$ items. The $n=1$ mean
completion length agreed to $0.4\%$, so the pipeline was sound and the *distribution* was not.
`results/onset_prediction_comma7b_seed.md` carries the full diagnosis. **That arm's band was never
computed**, so nothing here is chosen with knowledge of the answer.

**This arm changes exactly one thing from the arm on record: `--seeds 42 43 44` $\rightarrow$
`52 53 54`.** `--batch-size` is left at `h1.py`'s default of `8`, which is what
`output/phase5/sel_comma7b_64` used --- the launcher passes no `--batch-size` flag at all, so the
default is the protocol rather than a number we chose.

## What is measured

Held identical to `output/phase5/sel_comma7b_64`: the anchor (`common-pile/comma-v0.1-2t`), the same
$500$ ordinary prompts, `--trajectories-per-prompt 64`, `--max-new-tokens 200`, `--batch-size 8`
(the default, unpassed), the reward, both judges, and the seven-point $n$ grid. `build_trajectory_seeds`
hashes the `(prompt_id, seed, draw)` tuple into the high 16 bits, so the $64$ trajectory seeds are
**disjoint from the original $64$ by construction**.

Generation splits by prompt class across three cards and merges, as `feat-129`, `feat-131` and
`feat-132` did --- validated empirically, not argued: `feat-129`'s class-split pool reproduced the
single-card pool's $32{,}000$ rewards **bit-identically** at ranks $0$--$63$. Splitting does not
change the batch size within a class, so it is not a second re-roll; `feat-132`'s defect was the
batch size, not the split.

## The integrity check, rebuilt --- this is the other thing `feat-132` got wrong

Its gate was *"within `0.03` of the committed empty fraction."* Two defects, both now fixed:

1. **An absolute tolerance on a rate is not scale-free.** At KL3M-1.7B the quantity is
   $0.000$--$0.002$, so `0.03` is unfalsifiable there and could not have detected a batch effect of
   any size --- which is exactly why `feat-131` passed it without learning anything. At Comma-7B the
   quantity runs $0.030$--$0.094$ across three arms of the same model on the same prompts, so the
   same `0.03` is *tighter than the quantity's own arm-to-arm spread*.
2. **The aggregate hides the structure.** Under the corrected test below, `feat-132`'s **total**
   would have **passed** ($28$ against a pass band of $26$--$73$) while its **neutral** class still
   **failed** ($24$ against $26$--$68$). The check that carried the information is the per-class one,
   and the old gate did not have it.

**The committed check.** The reference counts are measured with the scorer's own code on
`output/phase5/sel_comma7b_64`, the arm being replicated --- never on a sibling arm at a different
$n$ (caution `(v)`):

| stratum | reference | passes for | as a rate |
|---|---|---|---|
| `neutral` | $45/200$ | $26 \le b \le 68$ | $0.130$--$0.340$ |
| total | $47/500$ | $26 \le b \le 73$ | $0.052$--$0.146$ |

Both must pass: a **two-proportion $z$ test at the $1\%$ level**, $|z| < 2.5758$, on the replication's
draw-0 empty counts against the reference's. It is scale-free, so the same rule is correct at an
anchor where the rate is $0.09$ and at one where it is $0.002$. The prompt count must be $500$.

Judged **levels** are not compared across passes (caution `(ap)`), and no number from this arm is set
against a number from any other sweep except through the **paired** difference below, computed within
one pass on both sides.

## There is deliberately NO bit-identity reproduction check

An independent draw is supposed to differ, so a gate of that shape would be incoherent rather than
merely wrong. This is `feat-131`'s reasoning and it is unchanged.

## Committed bands --- read on the paired $g(64)-g(8)$, judge~B, 500 prompts

The same three-row table as `feat-131` and `feat-132`, scored by the same code path, so all of these
arms are read by one rule rather than by three that look alike.

| reading | verdict |
|---|---|
| $>0$ with its interval excluding $0$ | **REPLICATES.** Comma-7B's climb to $n=64$ survives an independent draw. The two-anchor breadth-at-$n=64$ claim is verified rather than merely surviving, and the stability criterion has an out-of-sample confirmation at $2.43$ half-widths. |
| interval includes $0$ | **DOES NOT REPLICATE.** The climb at the second of the two retained anchors is a draw-level effect. The breadth-at-$n=64$ claim then rests on **the audited anchor alone**, and the appendix must say so in those words; the criterion is falsified out of sample and caution `(ap)` must record that a ratio above $2.4$ did not protect a reading. |
| $<0$ with its interval excluding $0$ | **INVERTS.** Stronger than non-replication. The appendix must then say the $n=64$ reading at this anchor is not stable in the seed, and the paper makes **no** across-anchor claim at $n=64$ at all. |

The reference band is `+0.1010 [+0.0590, +0.1420]`, $2.43$ interval half-widths from zero, against
TinyComma's $2.12$ (reproduced to four decimals) and KL3M-1.7B's $1.71$ (moved $0.061$). **The
stability criterion predicts REPLICATES**, which is the reason to run it: a rule that predicts the
answer can be wrong in public.

## Committed secondary, reported whatever it reads

$|D_{\text{rep}} - D_{\text{orig}}|$ against the two precedents ($0.0000$ at TinyComma, $0.0610$ at
KL3M-1.7B); the ratio $g$ / half-width in the replication; the full seven-point grid under **both**
judges; the per-class draw-0 empty counts whatever the gate says; and the judged levels, which are
*expected* to move between passes and are reported as information rather than gated on.

## Excluded in advance

We will not, after seeing results: change the scorer, either judge, the opponent, the prompt set, the
batch size or the $n$-grid; quote judge~A if judge~B disagrees; compare any judged *level* across
passes; re-run at a third seed and pick the two that agree; pool this arm with `feat-132`'s unread
one or with the arm on record; move the stability criterion's threshold to whichever side of $2.43$
the answer lands on; or treat a non-replication as a reason to drop the anchor rather than to narrow
the claim.

## What this arm cannot do, stated before it runs

It re-draws **one** anchor under **one** alternative seed triple. It says nothing about the two
Pleias anchors, nothing about KL3M-1.7B beyond what `feat-131` read, and nothing about where any
ceiling sits --- Comma-7B's ceiling past $n=64$ remains unmeasured. It cannot separate a reward-model
ceiling from an anchor-support ceiling. A replication that succeeds raises the warrant on the second
of two cells; it does not make two anchors a law, and the paper will keep saying the climb past $n=8$
is established at two anchors and not in general.

## Compute

Measured from the arm being replicated, at **the same batch size**, which is the point:
`results/compute_hours.csv` has `sel_comma7b_64` at **13.42 gpu-hours** for $32{,}000$ trajectories
on one card (2026-09-12 20:27 $\rightarrow$ 2026-09-13 09:52) and its scoring at **0.26 h**. Split by
class at that measured rate:

| card | class | trajectories | expected |
|---|---|---|---|
| GPU 1 | neutral $200$ | $12{,}800$ | $\approx 5.4$ h |
| GPU 2 | creative $150$ | $9{,}600$ | $\approx 4.0$ h |
| GPU 4 | factual $150$ | $9{,}600$ | $\approx 4.0$ h |
| GPU 1 | merge + score | --- | $\approx 0.3$ h |

**Total $\approx 13.7$ gpu-hours**, under the $24$-gpu-hour escalation threshold, at $\approx 5.7$
hours of wall clock. Batch $8$ is slower than the $32$ `feat-132` used and the estimate says so.
Checked with `env -u LD_LIBRARY_PATH nvidia-smi` before launch: GPUs 1, 2 and 4 idle at $14$ MiB;
**GPU 0 holds another user's $597$ MiB and is not taken**; GPU 3 is the 4 GB T400 and is never used.

## Scoring log
