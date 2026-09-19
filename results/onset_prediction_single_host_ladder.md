# Pre-registration: the complete non-Comma ladder, measured on ONE host

**feat-140.** Written and committed **before any of these arms runs**. Nothing above the
`## Scoring log` heading is edited afterwards.

## Why this arm exists

`feat-136`'s hypothesis **H2** --- *no non-Comma anchor clears $2.0$ interval half-widths* --- is the
paper's most consequential negative result, because it says the climb to $n=64$ is a property of the
Comma family rather than of selection anchoring in general. It currently rests on **six anchors
measured across two machines**: KL3M-1.7B, Pleias-1.2B and Pleias-3B on the local A100s, and
KL3M-170M, KL3M-520M and Pleias-350M on host B.

On `2026-09-19` this project measured, and recorded as caution (at), that comparing two arms produced
on different hosts **or through different pipelines** is a real confound rather than a theoretical
one --- it invalidated a host-transfer arm that night. A reviewer is entitled to ask whether H2 is an
artefact of which anchors happened to run on which machine. **This arm removes the question by
measuring every missing non-Comma anchor on host B**, so that the whole ladder exists on one
machine, one pipeline, one set of seeds.

## What runs

`scripts/run_breadth64.sh` **unmodified**, at its default `--seeds 42 43 44`, for the four anchors
absent from host B:

| anchor | model id | reading on record (local) | half-widths | verdict on record |
|---|---|---|---|---|
| Pleias-1.2B | `PleIAs/Pleias-1.2b-Preview` | $+0.0360$ | $0.95$ | SATURATED BY 8 |
| Pleias-3B | `PleIAs/Pleias-3b-Preview` | $+0.0030$ | $0.07$ | SATURATED BY 8 |
| KL3M-1.7B | `alea-institute/kl3m-003-1.7b` | $+0.0650\ [+0.0270, +0.1030]$ | $1.73$ | CLIMBS, **but see below** |
| KL3M-3.7B | `alea-institute/kl3m-003-3.7b` | none yet (`feat-135` is measuring it locally) | --- | --- |

## The committed predictions

**P1 (primary, this is H2 tested on one host).** **None of the four clears $2.0$ half-widths.**
H2 is falsified if any does, and that falsification is the finding, stated as such.

**P2 (per anchor).** Pleias-1.2B and Pleias-3B read **SATURATED BY 8**.

**P3 (KL3M-1.7B --- the one prediction that takes a side).** Its record carries a contradiction: the
original arm read `CLIMBS` at $+0.0650$ and $1.73$ half-widths, and a disjoint-seed replication read
$+0.0040\ [-0.0330, +0.0400]$, a move of $0.061$ --- the largest seed move ever measured here and the
source of the `MAX_SEED_MOVE` constant. Caution (ap) treats a $1.73$-half-width reading as MARGINAL
and the replication as the later evidence. **This arm predicts SATURATED BY 8**, siding with the
replication. If it reads CLIMBS again the prediction is wrong and that is recorded, not explained
away: two of three draws would then say CLIMBS and the anchor's status would be genuinely open.

**P4 (KL3M-3.7B).** No prior exists, so **no verdict is predicted**; only P1 applies. Inventing a
band for an anchor with no measurement would be a band chosen to be met. When `feat-135` lands
locally, the two become a host pair --- but that comparison belongs to whatever registers it, not
here.

## Integrity checks, fixed before the data exists

**I1 (BLOCKS) --- mean words GIVEN NON-EMPTY within $5\%$ of the local arm's.** Not the raw mean.
The same repaired statistic `feat-138` registers, for the same measured reason: on `2026-09-19`
`comma7bhb` failed a $5\%$ raw-mean tolerance at $+5.3\%$ and the decomposition showed the mean is
exactly `(1 - empty_frac) x (mean words given non-empty)`, with the **entire** failure in the empty
rate while length given non-empty moved $-2.7\%$. Caution (v) already recorded that an aggregate
gate on a stratified rate gates the wrong quantity. **Disclosed: that decomposition was known before
this check was written.** It is registered here before this arm's data exists, and is not applied
retroactively to any arm already gated.

**I2 (REPORTED) --- the empty fraction**, two-proportion $z$ against the local arm, per class and on
the total. Reported and never blocking across hosts: caution (as) established that at bf16 the
reduction order differs between machines and moves a rate-valued quantity, so a shift here is
expected rather than diagnostic.

**I3 (BLOCKS) --- coverage:** grid $\{1,2,4,8,16,32,64\}$ on all $500$ prompts, judge B present.

**I4 (BLOCKS) --- pipeline identity:** `pairing()` must read `target == anchor` on both the host-B
arm and the local arm it is set beside, which is the check caution (at) exists to impose.

## What this arm cannot do

It cannot address the **scale confound** in H2, and that confound must be stated plainly: the only
clean, public-domain-trained models at $7$B are the two Comma checkpoints, so "Comma climbs" and
"$7$B climbs" are not separated by any anchor that exists. What the ladder does offer against it is
the $\approx 1.7$B row --- TinyComma-1.8B climbs at $+0.0880$ while KL3M-1.7B and Pleias-1.2B do not
--- which is evidence at matched scale but from a single family pair, not a controlled comparison.

It also cannot certify that host B and the local box compute the same thing; caution (as) says
nothing can, at bf16. The distance $|D_{\text{host B}} - D_{\text{local}}|$ is **reported** against
the $0.0610$ range for context and is **not** a gate.

## Compute

Four arms, $500 \times 64$ at $1.2$--$3.7$B with `--max-new 200`, on four idle H100s. The three small
anchors of comparable size finished in $2.5$--$3$ wall-clock hours each on this host on
`2026-09-19`. Estimate **$\approx 12$ gpu-hours total, no single arm above $4$** --- under the
$24$-gpu-hour escalation threshold. Together with `feat-138`'s $\approx 11$ this is $\approx 23$
gpu-hours across eight cards in parallel, run under the user's instruction of 2026-09-20 to use host
B's GPUs to their maximum.

## Scoring log

### 2026-09-20 --- the prediction FAILED on all three scored anchors

| anchor | local, on record | host B | half-widths | predicted | read | distance |
|---|---|---|---|---|---|---|
| Pleias-1.2B | $+0.0360$ (SATURATED) | $+0.0560\ [+0.0180, +0.0940]$ | $1.47$ | SATURATED | **CLIMBS** | $0.0200$ |
| Pleias-3B | $+0.0030$ (SATURATED) | $+0.0730\ [+0.0320, +0.1130]$ | $1.80$ | SATURATED | **CLIMBS** | $\mathbf{0.0700}$ |
| KL3M-1.7B | $+0.0650$ (CLIMBS) | $+0.0470\ [+0.0110, +0.0810]$ | $1.34$ | SATURATED | **CLIMBS** | $0.0180$ |
| KL3M-3.7B | --- (none registered) | not finished | --- | none (P4) | --- | --- |

I1, I3 and I4 pass on all three: both sides self-paired on the same model id, and length given
non-empty agrees to $+0.4\%$, $+0.6\%$ and $+0.0\%$ --- so these are like-for-like comparisons and
not caution (at)'s defect.

**P2 failed twice and P3 once, and the two failures are different.**

* **P2.** Pleias-1.2B and Pleias-3B were registered as SATURATED and both **CLIMB** on host B.
* **P3.** KL3M-1.7B was registered as SATURATED, siding with the disjoint draw that moved $0.061$
  against its original CLIMBS. It **CLIMBS** again. This file said in advance what that means and it
  is not revised now: *"two of three draws would then say CLIMBS and the anchor's status would be
  genuinely open."* It is open.

**H2 SURVIVES its committed rule, and the rule is doing real work.** No non-Comma anchor clears
$2.0$ interval half-widths: the largest is Pleias-3B at $1.80$. Six non-Comma readings are now
scored across the two pre-registrations and not one is stable.

**But the finding must be restated, because "non-Comma anchors do not climb" is now false.** Three
of them do climb, marginally. What separates the families is **stability, not direction**: the Comma
readings sit at $2.07$--$2.43$ half-widths and reproduce ($0.011$, $0.013$ moves), while every
non-Comma reading sits below $1.81$ and moves enough to change its own verdict between draws.

**Set the two entries beside each other and the contrast is the result.** Same pipeline, same
anchors, one thing different:

| what changed | moves observed | verdicts changed |
|---|---|---|
| the seed, same host (`feat-138`) | $0.007$--$0.018$ | $0$ of $4$ |
| the host, same seeds (`feat-140`) | $0.018$--$0.070$ | $2$ of $3$ |

Pleias-3B's $0.0700$ is **beyond** the $0.0610$ that bounds every seed replication on record. A host
change is not a re-draw: it is larger, and at a marginal anchor it is large enough to flip the
verdict. That sharpens caution (as) --- which established that bf16 makes two hosts disagree on a
reward --- into a statement about a published quantity.

**What this cannot say, stated because the asymmetry is easy to miss.** There is **no unblocked
cross-host Comma comparison**: both Comma host-transfer arms are refused by `feat-136`'s G0b. So the
cross-host evidence here is entirely non-Comma, and the claim "Comma is stable across hosts while
the others are not" is **not** supported by these data. What is supported is that Comma is stable
across seeds and the others are not stable across hosts.
