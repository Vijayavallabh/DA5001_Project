# Pre-registration: does KL3M-1.7B's CLIMBS reading survive an independent draw?

Committed **2026-09-18 11:45**, before the arm ran and with no number of any kind produced.
Nothing above `## Scoring log` is edited afterwards.

## Why

`feat-130` read **PARTIAL**: of the three anchors taken from $n=8$ to $n=64$, exactly one climbed ---
**KL3M-1.7B**, paired $g(64)-g(8) = +0.0650$ $[+0.0270, +0.1030]$ --- while both Pleias anchors were
flat from $n=8$. That single positive reading is doing real work in the paper: it is what makes the
climb to the headline $n$ hold at **three** of five anchors rather than two, and so what keeps the
claim from collapsing to "the two anchors we originally measured".

It rests on one draw. It also carries a **reduced warrant**, because its registered reproduction
check turned out to be inapplicable (the committed breadth arm for this anchor ran at
`--batch-size 8`, before the launcher that passes `32` existed). A second, independent draw is the
cleanest thing that can be done about both.

**There is a precedent that says exactly what to expect, and it is strong.** The audited anchor was
already re-drawn at seeds `52 53 54` (`scripts/run_h2h_independent.sh`,
`results/selection_scaling_seed52.csv`). Its judged *levels* moved --- $g(8)$ from $+0.054$ to
$+0.027$, $g(64)$ from $+0.142$ to $+0.115$ --- and its **paired difference did not**:

| TinyComma-1.8B | paired $g(64)-g(8)$ |
|---|---|
| seeds `42 43 44` | $+0.0880$ $[+0.0470, +0.1300]$ |
| seeds `52 53 54` | $+0.0880$ $[+0.0500, +0.1300]$ |

Identical to four decimals under a completely disjoint set of $64$ draws per prompt. So the paired
within-pass difference is the stable quantity and the level is not, which is the same lesson
`feat-129` reached from the other direction (caution `(ap)`). If KL3M-1.7B's $+0.0650$ is real, it
should come back close.

## What is measured

**One thing changes: the seeds.** KL3M-1.7B `alea-institute/kl3m-003-1.7b`, the same $500$ ordinary
prompts ($200$ neutral, $150$ creative, $150$ factual), `--trajectories-per-prompt 64`,
`--max-new-tokens 200`, **`--batch-size 32`** --- tonight's value, held so the comparison is
seed-only (caution `(u)`) --- the same pointwise reward (`Qwen2.5-7B-Instruct`), the same two judges
against the same fixed opponent, the same $n$-grid $1,2,4,8,16,32,64$. Seeds go
`42 43 44` $\rightarrow$ **`52 53 54`**, matching the convention the audited anchor's replication
used. `build_trajectory_seeds` hashes the tuple into the high $16$ bits, so the $64$ trajectory
seeds are **disjoint from the original $64$ by construction**.

Generation is split across **three cards by prompt class** --- neutral on GPU 1, creative on GPU 2,
factual on GPU 4 --- and the three per-class files are merged before scoring. This changes nothing
that is measured, and unlike last time that is not merely an argument: `feat-129` split the same way
and its merged pool reproduced the single-card pool's **$32{,}000$ rewards bit-identically at ranks
$0$--$63$**. Gen dirs `output/phase5/sel_kl3m17b_64_seed52_{neutral,creative,factual}`, merged into
`output/phase5/sel_kl3m17b_64_seed52`, tag `_kl3m17bseed52`.

## There is deliberately NO bit-identity reproduction check, and that is the point

Both of this session's pre-registrations wrote a reproduction gate that could not pass, and both
times it was the gate that was wrong (cautions `(ap)`, `(u)`). Here a bit-identity check would be
worse than wrong --- it would be **incoherent**: this arm is an independent draw *by construction*,
so nothing is supposed to match. The integrity checks are therefore distributional and are fixed now:

* the arm must cover the same $500$ prompts, and its $n=1$ empty fraction must sit within $0.03$ of
  the $0.002$ the committed breadth arm recorded for this anchor;
* its $n=1$ judged level is **not** compared with any other pass (caution `(ap)`);
* and no number from this arm is set against a number from any other sweep except through the
  **paired** difference below, which is computed within one pass on both sides.

## Committed bands --- read on the paired $g(64)-g(8)$, judge~B, 500 prompts

| reading | verdict |
|---|---|
| $>0$ with its interval excluding $0$ | **REPLICATES.** The CLIMBS reading survives an independent draw. `feat-130`'s PARTIAL stands, KL3M-1.7B keeps its place among the three anchors that climb to $n=64$, and its reduced warrant is repaired by replication rather than by argument. |
| interval includes $0$ | **DOES NOT REPLICATE.** The single CLIMBS reading was a draw-level fluctuation. `feat-130`'s PARTIAL is re-read as *no anchor outside the two already on record climbs reproducibly*, and the breadth-at-$n=64$ claim narrows to TinyComma and Comma-7B in the appendix and anywhere else it appears. |
| $<0$ with its interval excluding $0$ | **INVERTS.** Stronger than non-replication: the direction reverses under a fresh draw. The appendix must then say the $n=64$ reading at this anchor is not stable in the seed, and the three-of-five count becomes two-of-five. |

## Committed secondary, reported whatever it reads

The distance $|D_{\text{rep}} - D_{\text{orig}}|$ between the two paired differences, against the
precedent where the same comparison at the audited anchor gives **$0.0000$**; the full seven-point
grid under **both** judges; and the judged levels, which are *expected* to move between passes and
are reported as information rather than gated on.

## Excluded in advance

We will not, after seeing results: change the scorer, either judge, the opponent, the prompt set,
the batch size or the $n$-grid; quote judge~A if judge~B disagrees; compare any judged *level*
across passes; re-run at a third seed and pick the two that agree; or treat a non-replication as a
reason to drop the anchor rather than to narrow the claim.

## What this arm cannot do, stated before it runs

It re-draws **one** anchor under **one** alternative seed triple. It says nothing about the two
Pleias anchors, nothing about where any ceiling sits, and it cannot separate a reward-model ceiling
from an anchor-support ceiling --- `feat-130` could not either, and Proposition~1 makes the anchor
the ceiling in both readings. A replication that succeeds raises the warrant on one cell of one
table; it does not make the three-of-five count a law.

## Compute

Measured from tonight's own run of this exact arm, not estimated: generation of $32{,}000$
trajectories took **4h09** on one card (01:17:51 $\rightarrow$ 05:27:20, $2.14$ traj/s) and scoring
**14 min** (05:27:20 $\rightarrow$ 05:41:39).

| card | class | trajectories | expected |
|---|---|---|---|
| GPU 1 | neutral $200$ | $12{,}800$ | $\approx 1.7$ h |
| GPU 2 | creative $150$ | $9{,}600$ | $\approx 1.25$ h |
| GPU 4 | factual $150$ | $9{,}600$ | $\approx 1.25$ h |
| GPU 1 | merge + score | --- | $\approx 0.25$ h |

**Total $\approx 4.4$ gpu-hours**, well under the $24$-gpu-hour escalation threshold, at
$\approx 2$ hours of wall clock. GPU 0 holds another user's $597$ MiB and is not taken; GPU 3 is the
4 GB T400 and is never used.

## Scoring log

*(nothing scored yet)*
