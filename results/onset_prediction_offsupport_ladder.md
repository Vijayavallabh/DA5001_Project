# Pre-registration: off the anchor's support, is selection beaten or just under-drawn? (feat-172)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why

feat-170 established that the headline reversal fails on AlpacaEval and holds on our own workload,
at both a vacuous and a binding budget, and the manuscript now scopes the claim to workloads inside
the anchor's support. **That scoping is stated more strongly than the evidence supports in one
specific way, and this arm is here to fix it.**

The appendix attributes the loss to a support ceiling: selection serves only what the anchor draws,
so on a workload the anchor cannot do, more draws cannot help. **The AlpacaEval ladder does not
show a ceiling.** Its gains are `+0.0000, -0.0081, +0.0031, +0.0180, +0.0335, +0.0634, +0.0764`
over `n = 1 ... 64`, and the last doubling still adds `+0.0130`. A curve still climbing at the
right edge of its grid is not evidence of a ceiling; it is evidence that we stopped measuring.

So the appendix's mechanism may be right and may be an artefact of `n = 64`. The distinction is
not cosmetic: the meter's binding-budget gain there is `+0.1174`, only `+0.0410` above selection's
`+0.0764`, which is about three doublings at the observed rate. **If selection catches a binding
meter off-support at `n = 256` or `512`, then the paper's own scoping concession is too strong ---
and it would be caught at a certificate of `\log 512 = 6.24` nats against the meter's realised
`171.3`, still a factor of `27`.**

## What runs

One generation: the anchor (`TinyComma-1.8B`, `k=0`) on the same `805` AlpacaEval prompts at
`--trajectories-per-prompt 256`, `--batch-size 64`, every other flag identical to feat-166's
`sel_anchor64`. **One run yields `n = 64`, `128` and `256`**, because `h1.py` seeds per trajectory
index and feat-134 demonstrated the prefix property directly (its first `64` of `128` draws were
bit-identical to the committed `64` in all `32{,}000` rewards).

Then the reward pass over the new candidates and `analysis/selection_scaling.py` under judge~B.
The metered arm, the opponent and the `n <= 64` rewards are **not** re-run.

## The reproduction gate --- on the reward cache, never on a judged number

Ranks `0`--`63` of this arm's reward cache must be **bit-identical** to
`results/mixpow_rewards64.csv`, all `51{,}520` floats compared with `==`. This is feat-134's gate,
which passed at `32{,}000` of `32{,}000`, and it is the only thing that licenses treating the new
draws as an extension of the committed ladder rather than a fresh pool.

**No `n > 64` number is read until it clears.** If it fails the reading is INAPPLICABLE rather than
a failure of the arm: it would mean the batch size or the seeding differs and the two pools are
different draws, and the response is to chase that, not to read the ladder anyway.

## Bands, committed before the data

- **B1 --- does selection catch the binding meter off-support?** Judge~B's paired `D3` against the
  `k=1.0` metered arm at each of `n = 128` and `256`. **CATCHES** if `D3 >= 0` at either, with the
  interval containing or above zero; **CLOSES** if `D3` is still negative but its magnitude at
  `n=256` is less than half the `-0.0339` at `n=64`; **CEILING** if the magnitude does not fall by
  at least a quarter. Reported with the `n` at which each happens.
- **B2 --- where does the ladder stop?** The paired `g(128) - g(64)` and `g(256) - g(128)` within
  this pass. **STILL CLIMBING** if the latter is `> 0` with its interval excluding zero;
  **SATURATED** if it contains zero. This is feat-134's construction and is the one that speaks to
  the ceiling directly.
- **B3 --- the price of catching up.** Whatever B1 says, report the certificate at the crossing or
  at the grid's end: `\log n` nats against the metered arm's realised spend on this corpus,
  as a ratio. A selection arm that needs `256` draws still publishes `5.55` nats.

## Excluded in advance

- Extending past `n = 256` inside this arm after seeing where the curve goes. If the answer is "it
  would cross at `512`", that is a **separate** registration with its own bands.
- Reading any band if the reproduction gate fails.
- Comparing a judged level from this pass to one from any other (caution (ap)); B1 and B2 are
  paired differences within this pass, and the metered arm it is set against is judged in the
  same pass.
- Presenting a CATCHES reading as rescuing the headline. The headline is a claim about `n = 64`;
  if it takes `256` draws off-support, the honest statement is that the workload costs selection
  two doublings, and the scoping concession is **rewritten, not deleted**.

## What we predict

**CLOSES, not CATCHES.** We expect the gap to narrow and not to close by `n = 256`: extrapolating
`+0.0130` per doubling gives about `+0.102` at `n=256` against the meter's `+0.1174`. We predict
B2 reads STILL CLIMBING at `128` and is marginal at `256`. **We are predicting that our own
appendix's "ceiling" language is wrong and its conclusion is right** --- the reversal really does
fail off-support at the `n` the paper uses, but not because more draws cannot help.

## Compute

Host B, one card, about `11` hours for the generation, `40` minutes for the rewards, an hour for
judging.

## Scoring log

## Arm B, registered 2026-09-22 12:20 --- the ON-support control, before it runs

**Why the arm as registered is not enough.** B2 asks whether the AlpacaEval ladder is still
climbing at `n = 128` and `256`. Whatever it says, that reading alone cannot support the sentence
it is meant to inform. If the off-support ladder climbs, the honest question is immediately
*"compared to what?"* --- our own workload's ladder must be measured on the **same pipeline, the
same `n` grid and the same prompt count**, or the comparison is against feat-129's `500`-prompt
`n <= 128` pass at a different batch size, which is exactly the cross-pass comparison this project
keeps getting wrong (cautions (u), (v), (ap)).

**What runs.** The anchor on our own `850` ordinary prompts at `--trajectories-per-prompt 256`,
`--batch-size 64`, sharded by prompt class across the two free local cards --- `neutral` + `creative`
(`350`) on one and `factual` (`500`) on the other, the sharding `scripts/run_wscope.sh` already
uses and which feat-170 Arm A validated (its three shards reassembled to exactly `200`/`150`/`500`).
Every other flag matches feat-170 Arm A, whose `n <= 64` draws this extends.

Runs **locally**, on GPUs `2` and `4`: those are the only free cards here (`0` and `1` hold another
user's `65` and `62` GB, `3` is the `4` GB T400). Host B is running Arm A of this registration on
card `0` and MT-Bench on card `1`, with its other six cards taken by another user of that shared
account.

**Gate.** Ranks `0`--`63` of this arm's reward cache must be bit-identical to
`results/wscope_rewards64_a.csv`, all `54{,}400` floats compared with `==`. Same construction as
Arm A's gate and feat-134's; **no `n > 64` number is read until it clears**, and a failure means
the pools are different draws and is INAPPLICABLE rather than a result.

**B5 --- the ladder shape, on support.** The paired `g(128) - g(64)` and `g(256) - g(128)` within
this pass, judge~B. **STILL CLIMBING** if the latter is `> 0` with its interval excluding zero,
**SATURATED** if it contains zero.

**B6 --- the comparison B2 needs.** The two ladders' shapes set side by side, each measured within
its own pass and never by comparing levels across them (caution (ap)). The claim the manuscript may
then make is about **shape**: if the on-support ladder saturates by `64` and the off-support one is
still climbing at `256`, then "selection has hit its ceiling off support" is **false** and the
opposite of what our appendix said this morning; if both saturate, the ceiling language is right
and only its reason was wrong; if both climb, `n = 64` is simply not where either ladder ends and
the paper's choice of `64` is the thing to defend.

**Excluded in advance:** reading B5 or B6 if the gate fails; extending past `256` inside this arm;
and quoting `g(256)` from this pass beside `g(256)` from Arm A's --- the shapes are compared, the
levels are not.

**We predict SATURATED on support and STILL CLIMBING off it**, which is the combination that makes
our own "ceiling" sentence wrong in the most interesting way: the ceiling would be a property of
the workload where selection is *already winning*, not of the one where it loses.

**Compute.** Local, two cards, about `9` hours for the larger shard.

### Arm B was launched on the wrong host, and could never have passed its own gate

Caught 2026-09-22 16:22, when the `small` shard died of CUDA OOM --- the sibling project grew to
`56.76` GB on the card it shared. Diagnosing the OOM surfaced the larger error.

**Arm B's gate requires ranks `0`--`63` bit-identical to `results/wscope_rewards64_a.csv`. That
cache was generated and scored on host B. Arm B was launched on the local host.** feat-136 measured
what that costs: across hosts only **`17.6%`** of rewards agree to `1e-2` and the served completion
changes on `5.1%` of cells, because a different bf16 reduction order gives different text and
different scores. Bit-identity across hosts is not merely unlikely, it is something feat-136
already established cannot happen.

So the arm was doomed at launch and the OOM saved about nine hours of it. **The registration named
the right gate and the launch ignored what the gate implies about where the arm must run** --- the
`## Compute` line even says "Local host, GPUs 2 and 4", written in the same session that wrote the
gate. Per caution (w) this is a defect in our own specification, the arm is INVALID rather than
failed, and it is re-run on host B.

**The rule, which caution (at) half-stated and this completes:** a bit-identity gate against
another arm's cache is a constraint on the **host**, not only on the flags. Caution (at) says
derive the reference from the arm being replicated and assert the pipelines match; the pipeline
includes the silicon, and feat-136 measured exactly how much. Any arm whose gate is `==` must run
where its reference ran.

**Also recorded: caution (c), tenth incident, in the cleanup.** The kill used
`ps -eo pid,args | awk '/h1\.py/ && /cap-factual 500/ && /256/'`, and the invoking shell's own
command line contained all three patterns, so awk returned the shell's PID and killing it killed
the shell (exit `144`). The caution's own prescription --- match on the **executable field**,
`$2 ~ /python$/` --- was not followed. Both jobs did stop and no CUDA child was orphaned, checked
by `nvidia-smi --query-compute-apps`.

### The gate, read 2026-09-23 before any judged number existed --- PASS on both arms

The reward passes ran on host B, where both references were scored (Arm A GPU 0, Arm B GPU 6,
`scripts/run_offonsup_rewards.sh`, `--rewards-only` at the committed batch size `16`), each in about
`33` minutes. `scripts/run_offonsup_judge.sh` re-checked the registered gate with
`analysis/score_n128.py:reward_gate` before judging anything:

- **Arm A**: ranks `0`--`63` of `offsup_rewards256.csv` equal `mixpow_rewards64.csv` exactly ---
  `51{,}520` of `51{,}520` rewards compared with `==`. **PASS.**
- **Arm B**: ranks `0`--`63` of `onsup_rewards256.csv` equal `wscope_rewards64_a.csv` exactly ---
  `54{,}400` of `54{,}400`. **PASS.** Arm B was re-run on host B after its first launch was found
  to be on the wrong host (above); on the right host its gate is satisfiable and it is satisfied.

So both n=256 pools are the committed ladders extended, not fresh draws, and the bands may be read.
The judges started at 03:14 (host time), behind the gate by construction.
