# Pre-registration: TokenSwap's auxiliary is a stated requirement, not a free parameter (feat-167)

Committed **before any rung is run**. Nothing above the `## Scoring log` line is edited afterwards.

## Why

feat-165 runs TokenSwap with `TinyComma-1.8B`, chosen because it is already vetted at `0.000` and
shares Llama-3's tokenizer. It is also roughly `22x` larger than the `DistilGPT-2` their paper
uses, so **every reading feat-165 produces is at an auxiliary more capable than the method asks
for**. That cuts two ways and the arm cannot tell them apart: a bigger auxiliary should cost less
utility, and it may also memorise more.

This is the same question this paper already asked about its own scorer and answered with a ladder
(Appendix~A.1, `0.494`B to `72.7`B). The mechanism's requirement is a cost, and a requirement whose
size is unmeasured is a hidden parameter.

**And it settles, for their own configuration, the thing Appendix~J said had to come first.** The
sentence was: the auxiliary's *"own contamination would have to be vetted before any leakage number
from it meant anything"*. `DistilGPT-2` descends from GPT-2, trained on WebText, which is scraped
from outbound Reddit links and is not an openly licensed corpus. Nobody has run our vetting check
on it. We will.

## What runs

**Half A --- vetting, four auxiliaries, the protocol Appendix~D already uses for anchors.**
Near-verbatim recall of `50`-token windows on the `100` protected passages at a `100`-token raw
prefix, each model alone, no defence. Candidates: `DistilGPT-2` (`82`M, theirs), `KL3M-170m`,
`Pleias-350m-Preview`, `TinyComma-1.8B` (the rung feat-165 used, re-run here so the ladder is one
pass).

**Half B --- the utility rung at each auxiliary.** `analysis/tokenswap_decode.py --arms tokenswap`
on the same `850` ordinary prompts with `--chat`, the flags `scripts/run_memfree.sh` uses. The
rule-off control is not re-run: it does not depend on the auxiliary and feat-165 generates it once.
`G` is paired across vocabularies where they differ, by the mapping their paper relies on --- a
surface form counts only if it is a single token in **both**, and the count that survives is
reported per rung.

## Gates

- **G0 (the pairing survives).** At most `20` of the `110` words may fail to pair into a given
  auxiliary's vocabulary. Above that the rule being run is materially weaker than the one the
  authors specify and that rung is reported as NOT RUN rather than as a weak result.
- **G1 (the rule binds at every rung).** `changed_frac > 0.01`, as in feat-165.

## Bands

- **B1 --- does their own auxiliary pass our vetting check?** Near-verbatim recall of `DistilGPT-2`
  alone on the `100` passages. **CLEAN** at `<= 0.01`, **LEAKS** above it. The same reading is
  reported for all four.
- **B2 --- does suppression depend on the auxiliary?** feat-165 reads `0.0000` recall and `0`/`100`
  at ROUGE-L `>= 0.5` with the `1.8`B rung. Any rung that reads above `0.05` recall makes
  suppression auxiliary-dependent, which the paper would then have to state.
- **B3 --- does utility?** The judged order-averaged gain at each rung against the shared rule-off
  control. Reported as a series with its shape named, the way the scorer ladder is
  (MONOTONE / NOT MONOTONE, and where the peak is).

## Excluded in advance

- Dropping a rung because it is unflattering to TokenSwap or to us.
- Treating a `DistilGPT-2` leakage reading as a criticism of their *method*: it is a property of
  the auxiliary, and the method's own text says the auxiliary is a choice. What it would criticise
  is running the method without checking that choice --- which is what our own Appendix~J said.
- Re-running the control per rung and comparing rungs against different controls.

## What we predict

`DistilGPT-2` **LEAKS** is genuinely uncertain: at `82`M it is small enough that Appendix~D's own
finding --- that small openly licensed anchors read `0.000` --- may hold for it too, and its
training corpus is the only reason to doubt it. We predict **CLEAN**, at `<= 0.01`, and note that
predicting the comfortable outcome is exactly the case where a committed band matters. On B2 we
predict suppression holds at every rung, since a smaller auxiliary memorises less. On B3 we
predict the gain falls as the auxiliary shrinks, and that the `82`M--`350`M rungs cost more than
the `1.8`B one.

## Compute

Host B, four cards. Half A about `40` minutes. Half B about `90` minutes per rung; the
cross-tokenizer rungs are slower because the auxiliary must be re-encoded from text at every step.

## Scoring log

## Scoring, 2026-09-22

Half A was scored on 2026-09-21 (all four auxiliaries **CLEAN**). This is Half B, plus band **B2**,
which the registration named and no declared run produced --- see the note under B2 below.

### Gates

| rung | params | words unpaired of `110` | `|G|` token ids | binds (utility) | G0 | G1 |
|---|---|---|---|---|---|---|
| `DistilGPT-2` | `82`M | `0` | `426` | `28.89%` | **PASS** | **PASS** |
| `KL3M-170m` | `170`M | `5` | `171` | `3.07%` | **PASS** | **PASS** |
| `Pleias-350m` | `350`M | `0` | `397` | `25.26%` | **PASS** | **PASS** |
| `TinyComma-1.8B` (feat-165) | `1.8`B | `0` | `431` | `17.90%` | **PASS** | **PASS** |

Every gate passes at every rung, and **G0 passing at `KL3M-170m` is a defect in G0, stated here
before any band is read.** G0's own registered sentence gives its purpose --- above the threshold
*"the rule being run is materially weaker than the one the authors specify"* --- and it put the
threshold on WORDS. Only `5` of `110` words fail to pair at `KL3M`, so the gate passes; the rule
that survives holds `171` of `431` token ids and `1.01%` of the main model's probability mass
against `37.3%`, and binds on `1.79%` of leakage steps against `25.27%`. The quantity G0 was for is
the mass or the surface-variant count, and the quantity it measured is the word count. A word
survives if *any* of its surface forms is a single token in both vocabularies; the rule is carried
by *all* of them, and `KL3M` keeps a third.

G1 is worse, because G1 did not merely fail to fire --- it **certified** the rung. `1.79% > 1%`, so
a rule too weak to protect anything passed a gate whose job is to establish that the rule is
active. Caution (as) says a gate nothing can pass gates nothing; this is the mirror, a gate almost
anything passes. Neither threshold is moved now that the data are in (caution (as) again): they are
reported as passed, with their stated purpose recorded as unmet, and the rungs are read with the
bind rate printed beside every number so no reading can be quoted without it.

### B1 --- their own auxiliary passes our vetting check

Scored 2026-09-21. `DistilGPT-2` reads `0.0000` near-verbatim recall and `0`/`100` at ROUGE-L
`>= 0.5`, as do `KL3M-170m`, `Pleias-350m` and `TinyComma-1.8B`. **CLEAN at every rung**, which is
what we predicted and said was the case where a committed band matters. Appendix J's stated blocker
--- that the auxiliary's own contamination would have to be vetted before any leakage number from
it meant anything --- is cleared for their configuration, by measurement.

### B2 --- suppression IS auxiliary-dependent, and the axis is not size

**The band was unmeasurable from the runs the registration declared.** "What runs" named Half A
(the auxiliary alone, un-defended --- that is B1) and Half B (utility). Neither produces a
per-rung leakage number under the swap rule. Per caution (w) a defect in our own specification must
not retire a question, so the missing arm was run (`scripts/run_ts_rung_leak.sh`, feat-165's
leakage flags verbatim, only the auxiliary changed, against the same shared control).

| rung | params | `|G|` ids | mass on `G` | binds | nv-recall | `lcs_word` | ROUGE-L `>= 0.5` | `>= 0.3` |
|---|---|---|---|---|---|---|---|---|
| the memoriser alone | --- | --- | --- | --- | `0.3761` | `67.62` | `45`/`100` | `69`/`100` |
| `DistilGPT-2` | `82`M | `426` | `0.3732` | `29.46%` | `0.0000` | `2.92` | `0`/`100` | `0`/`100` |
| **`KL3M-170m`** | `170`M | `171` | `0.0101` | `1.79%` | **`0.2113`** | `41.15` | **`23`/`100`** | **`50`/`100`** |
| `Pleias-350m` | `350`M | `397` | `0.3848` | `27.79%` | `0.0000` | `2.86` | `0`/`100` | `0`/`100` |
| `TinyComma-1.8B` | `1.8`B | `431` | `0.3745` | `25.27%` | `0.0000` | `2.92` | `0`/`100` | `0`/`100` |

`0.2113` is far above the band's `0.05`, so **B2 fires and the paper has to state it.** But it
fires on an axis the band did not anticipate and which the registration's prediction got backwards.
We predicted suppression would hold at every rung *"since a smaller auxiliary memorises less"* ---
reasoning about capability. The rung that fails is neither the smallest nor the largest: it sits
**between** two rungs that suppress completely, at `170`M, with `82`M and `350`M both at `0.0000`.
Capability does not order this at all.

What orders it is whether the auxiliary's tokenizer can represent `G`. `KL3M` keeps `171` of the
`431` token ids and `1.01%` of the mass, so the rule has almost nothing to act on and the memoriser
is served essentially unmodified on `98%` of steps; recall lands at `0.2113`, between full
suppression and the un-defended `0.3761`, exactly where a partially-applied rule should land. The
reading is not that `KL3M` is a leaky auxiliary --- alone it is `0.0000` (B1) --- it is that
**TokenSwap's guarantee is a property of the (`G`, auxiliary tokenizer) pair and degrades silently
when they are mismatched.** Nothing in the run announces it: the rule loads, reports a live bind
rate, passes our own activity gate, and returns `23`/`100` verbatim passages.

That is the finding of this arm, and it is a criticism of deploying the method without checking the
pairing --- which is the same criticism our own Appendix J levelled at using an unvetted auxiliary.
It is not a criticism of the method as its authors specify it: they use `DistilGPT-2`, which pairs
at `426`/`431` and suppresses completely.

### B3 --- utility, with the bind rate printed beside it

Shared rule-off control (`norule`), the one every rung is read against: `+0.2720` over the anchor,
`n = 500`. `analysis/ts_rung_ladder.py`.

| rung | params | `|G|` ids | binds | gain | cost vs the shared control | 95% CI |
|---|---|---|---|---|---|---|
| `DistilGPT-2` | `82`M | `426` | `28.89%` | `+0.0740` | `-0.1980` | `[-0.2225, -0.1745]` |
| `KL3M-170m` | `170`M | `171` | `3.07%` | `+0.2420` | `-0.0300` | `[-0.0505, -0.0100]` |
| `Pleias-350m` | `350`M | `397` | `25.26%` | `+0.1030` | `-0.1690` | `[-0.1920, -0.1460]` |
| `TinyComma-1.8B` | `1.8`B | `431` | `17.90%` | `+0.1680` | `-0.1040` | `[-0.1255, -0.0820]` |

Over all four rungs the series is **NOT MONOTONE**, peaking at `KL3M-170m`. That shape is an
artefact and must not be reported as the answer: `KL3M`'s rung is cheap because its rule is absent,
and it is the same rung that leaks `23`/`100`. Over the three rungs whose rule binds within `4x` of
the median the series is **MONOTONE RISING in auxiliary size** --- `+0.0740` (`82`M), `+0.1030`
(`350`M), `+0.1680` (`1.8`B) --- which is the registration's prediction, *"the gain falls as the
auxiliary shrinks, and the `82`M--`350`M rungs cost more than the `1.8`B one"*, **CONFIRMED**. The
exclusion is mechanical (a bind rate an order of magnitude off the others) and is applied by the
scorer, not by hand.

### What this does to feat-165's headline

feat-165 reported `TokenSwap - selection = +0.0615 [+0.0285, +0.0950]`, **INCUMBENT WINS**, at
`TinyComma-1.8B` --- an auxiliary `22x` larger than the one TokenSwap's own paper uses, which that
arm's own text flagged as the reason this ladder had to be run. At `DistilGPT-2`, their
configuration, the same paired comparison reads:

| auxiliary | `TokenSwap - selection`, paired | reading |
|---|---|---|
| `DistilGPT-2` (`82`M, **theirs**) | `-0.0325 [-0.0645, +0.0005]` | **TIE** |
| `Pleias-350m` (`350`M) | `-0.0035 [-0.0375, +0.0305]` | **TIE** |
| `TinyComma-1.8B` (`1.8`B, feat-165's) | `+0.0615 [+0.0285, +0.0950]` | INCUMBENT WINS |

So **TokenSwap beats selection anchoring only at an auxiliary `22x` larger than the one its own
paper specifies; at their configuration the two tie.** feat-165's `INCUMBENT WINS` is not withdrawn
--- it is a correct reading of the rung it was measured at, and that rung is the one that gives
TokenSwap its best case. What is withdrawn is any statement of it without the auxiliary attached.
The `-0.0325` interval's upper end is `+0.0005`, which is a tie by a hair and will be reported as a
tie, not as a selection win.

None of this touches suppression, where TokenSwap at `DistilGPT-2` is still total (`0`/`100` at
both thresholds) and still stronger than MemFree (`5`/`100` at `>= 0.3`), nor the cost ordering:
selection remains the most expensive of the three mechanisms and the only one publishing a bound on
the served law.

### Excluded in advance, and honoured

No rung was dropped. `KL3M-170m` is the rung that most embarrasses both the method and our own
gates, and it is reported in full, in every table, with the mechanism named. `DistilGPT-2`'s
leakage reading is reported as a property of the auxiliary, not of the method. Every rung is read
against the one shared control; none was re-run per rung.
