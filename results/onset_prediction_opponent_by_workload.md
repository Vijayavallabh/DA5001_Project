# Pre-registration: is the workload split the OPPONENT's strength? (feat-178)

Committed **before any generation and before the reanalysis in P1 is computed**. Nothing above the
`## Scoring log` line is edited afterwards.

## Why this arm exists

Two axes have been measured this week and neither has been set against the other.

**The workload axis** (feat-166/168/170/173/174/176): selection's advantage over a rate-matched
metered decoder is positive on prefix-completion workloads --- ours `+0.0645`, Gutenberg `+0.0990`
--- and negative or absent on instruction-following and reading comprehension --- AlpacaEval
`-0.0339`, MT-Bench `-0.0875`, CoTaEval-QA `+0.0070` and `-0.0375`. feat-176's own registration
says four points do not identify a cause and that a mechanism would have to come from a designed
intervention rather than from more benchmarks.

**The opponent axis** (feat-175): on ONE workload --- ours --- the same comparison falls
monotonically with the strength of the fixed opponent it is judged against. Judge~B reads
`+0.0645`, `+0.0490`, `+0.0195`, `-0.0200`, `-0.0065` over opponent strengths `0.555`, `0.704`,
`0.773`, `0.825`, `0.850`, Spearman `-0.900`; judge~C agrees on the shape and crosses earlier.

**These may be the same axis.** Every external benchmark is judged against the same fixed opponent,
`Llama-3.1-8B-Instruct`, which is an INSTRUCTION-TUNED model: strong where the task is to follow an
instruction and weak where it is to continue a passage. If so, "task type" is a proxy and the real
variable is how good the opponent happens to be at the workload --- which would be a mechanism, and
would also sharply limit what the workload result means.

## The quantity, and why it is comparable across passes at all

**Opponent strength** is `1 - mean(u_anchor_k0)`: how often the fixed opponent beats the anchor's
own `n=1` completion, order-averaged, on that workload's own prompts. It is a judged LEVEL, and
caution (ap) forbids setting judged levels from different sweeps against each other --- with one
measured exception this arm depends on. **Order averaging judges both orders and so draws nothing
from the presentation-order RNG**, which makes it a deterministic function of the text under a
greedy judge; `results/n128_order_averaged_note.md` measured a case where a single-order gain moved
`+0.142 -> +0.076` on byte-identical text while the order-averaged reading reproduced to four
decimals. Every number here is order-averaged for that reason, and no single-order level is quoted.

It is **workload-relative by construction**, and that is the hypothesis rather than a defect: the
claim under test is precisely that the same opponent is strong on one workload and weak on another.

## What runs

**P1 --- the reanalysis, on data already committed, no GPU.** `u_anchor_k0` is a column of every
`results/order_averaged_h2h_per_prompt__*.csv` already on disk, so opponent strength is measurable
for all six workload arms at no cost: ours, AlpacaEval, MT-Bench, Gutenberg, CoTaEval-QA and
(when it lands) unseenbooks. Written to `results/opponent_by_workload.csv` by
`analysis/opponent_by_workload.py`.

**L --- the designed intervention.** Run feat-175's ladder ON ALPACAEVAL. Five opponents ---
`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-{0.5B,1.5B,3B,14B}-Instruct` --- each generating
one completion for every AlpacaEval prompt, **all five through one generator**
(`analysis/blocklist_decode.py --arms plain --chat --max-new 200 --temperature 1.0 --seed 1234`,
`--data-dir data/bench/alpaca --split ordinary`). The committed ladder mixes two generators and
records which; this one does not, so its slope is measured inside one pipeline (caution (at)).

Then `analysis/order_averaged_h2h.py` under judge~B at the binding budget, changing **only
`--baseline-dir`** from `scripts/run_mixpow_judge_k.sh`: selection arm `output/mixpow/sel_anchor64`,
metered arm `output/mixpow/conc_k10` (which holds `k=1`, the binding cell --- the directory names
in that tree are swapped with respect to their contents), anchor control the same selection
directory, rewards `results/mixpow_rewards64.csv`, `--n 64 --k 1 --seed 7717`.

## Gates, read in this order, before any band

- **G0 (one pipeline)**: all five opponent directories carry the same generator, the same decoding
  flags and the same prompt count. A rung generated differently is dropped, not compared.
- **G1 (one prompt set)**: every rung's judged intersection with the selection arm is the same set
  of AlpacaEval prompts, and its size equals the committed pass's `805`.
- **G2 (no degenerate opponent)**: each opponent's empty-completion fraction is under `10%`, and
  its mean completion length is within a factor of `2` of the median rung's. An opponent that
  emits nothing wins nothing, and that is an artefact rather than weakness.
- **G3 (the instrument has range HERE)**: the five measured AlpacaEval strengths must span at least
  `0.10`. If they do not, the ladder has no leverage on this workload and the arm is
  **NOT TESTED** --- feat-175's own outcome when no reachable model was weaker than the committed
  opponent, and the honest label for an instrument with no range. It is not a refutation.

## Bands, committed before the run

**L1 --- does the slope reproduce off our corpus?** Spearman between AlpacaEval-measured opponent
strength and AlpacaEval `D3`, over the five rungs.

| reading | band |
|---|---|
| **CONSISTENT** | `rho <= -0.7` |
| **REFUTED** | `rho >= +0.3` |
| **UNRESOLVED** | between them |

Exact permutation `p` is reported. At `n=5` the smallest attainable two-sided `p` is `0.0167`, so
**no significance claim is made from this arm**, whatever `rho` reads.

**L2 --- does AlpacaEval ever cross?** The decisive reading.

| reading | band |
|---|---|
| **OPPONENT EXPLAINS THE SPLIT** | some rung reads `D3 > 0` with its 95% interval excluding zero |
| **TASK TYPE SURVIVES** | no rung crosses, and G3 passed so the ladder had range |
| **NOT TESTED** | G3 failed |

**L3 --- P1's table.** Opponent strength per workload against the committed opponent, beside each
workload's `D3`, and the Spearman over the six. Same three-band rule as L1. This is a reanalysis of
committed arms and is reported as one: it can suggest the unification and it cannot establish it,
because strength and task type are themselves correlated across these six workloads and no
reanalysis can separate two variables that move together in the data it is given.

**L4 --- the arithmetic check, registered because it could explain L1 and L3 away.** `D3` is a
difference of win rates against the same opponent, so it has less room when `u_anchor_k0` is near
`0` or `1`. Report `D3` divided by the available headroom `min(u_anchor_k0, 1 - u_anchor_k0)` and
re-run L1's Spearman on it. If the ordering survives the normalisation it is not a ceiling effect;
if it does not, say so and withdraw the unification claim.

## What each outcome does to the manuscript, fixed now

- **OPPONENT EXPLAINS THE SPLIT.** The appendix's task-type scoping is rewritten as an
  opponent-strength scoping, which is a stronger and more useful statement for a deployer: it says
  when selection helps in terms of the baseline being compared against rather than in terms of a
  benchmark's genre. The workload paragraphs stay, re-read through it.
- **TASK TYPE SURVIVES.** The two axes are independent, the appendix says so, and feat-175's
  paragraph gains the sentence that its ladder is a property of our workload and does not transfer.
- **NOT TESTED.** Reported as an instrument failure with its span, exactly as feat-175's H1 was.

**We predict CONSISTENT and TASK TYPE SURVIVES** --- that the slope reproduces (the ladder is
measuring something real about opponents) but that AlpacaEval does not cross even at the weakest
rung, because its committed `D3` is `-0.0339` and feat-175's ladder spans only about `0.07` of
`D3` end to end. If it crosses, the task-type sentence in the appendix is the one that has to go.

## What may not be claimed

- No significance from `n=5` or `n=6` rank correlations. The bands are descriptive thresholds.
- No causal claim from P1/L3, which are reanalyses of arms run for other reasons.
- No judged level quoted across passes except the order-averaged `u_anchor_k0` this document
  defines, and never a single-order one.
- Nothing about judge~C or Mixtral here. Judge~B only, as the committed AlpacaEval pass used.

## Excluded alternatives

- Adding a sixth rung after seeing the five.
- Re-running any rung at a different budget after seeing its `D3`.
- Dropping a rung because its point is inconvenient; G0/G2 are the only reasons a rung leaves, and
  both are about the generation rather than the reading.
- Pooling this ladder with feat-175's. Different workload, different prompt set; they are reported
  side by side and never averaged.

## Compute

Host B, the cards the sibling project and the in-flight arms are not using. Five generations of
about `1{,}155` prompts each (AlpacaEval's `805` plus the `350` committed ordinary prompts the
bench directory's other two slots carry --- the judge intersects to the `805`, and the extra
`30%` is cheaper than building a sixth corpus directory), then five judge passes.

## Scoring log

### P1/L3/L4 SCORED, 2026-09-22 23:45 --- CONSISTENT, with one inversion that matters

Five workloads at their rate-matched **binding** budgets (the cell the workload comparisons are
primary on; mixing a binding cell with a vacuous one would compare two mechanisms). unseenbooks is
still generating and enters this table when it lands.

| workload | task type | n | opponent strength | `D3` | `D3`/headroom |
|---|---|---|---|---|---|
| Gutenberg | completion | `500` | `0.4865` | `+0.0990` `[+0.0795, +0.1185]` | `+0.2035` |
| CoTaEval-QA | comprehension | `500` | `0.5685` | `+0.0070` `[-0.0175, +0.0320]` | `+0.0162` |
| ours | completion | `850` | `0.5735` | `+0.0965` `[+0.0765, +0.1162]` | `+0.2263` |
| MT-Bench | instruction | `80` | `0.6594` | `-0.0250` `[-0.0781, +0.0281]` | `-0.0734` |
| AlpacaEval | instruction | `805` | `0.6792` | `-0.0339` `[-0.0534, -0.0137]` | `-0.1057` |

**L3: `rho = -0.900`, exact `p = 0.0833` over all `120` permutations --- CONSISTENT.**
**L4: normalised by the available headroom, `rho = -0.700`, `p = 0.2333` --- CONSISTENT**, and
exactly on the registered threshold. So the ordering is not an arithmetic ceiling effect, and the
normalised reading is the weaker of the two; both are quoted and neither is a significance claim
(at `n=5` the smallest attainable two-sided `p` is `0.0167`).

**The inversion is the finding, not the correlation.** CoTaEval-QA has almost exactly our own
workload's opponent strength --- `0.5685` against `0.5735` --- and a `D3` an order of magnitude
smaller, `+0.0070` against `+0.0965`. A pure opponent-strength account predicts those two
workloads look alike and they do not. So opponent strength orders the five and does **not** predict
the magnitude, and whatever the AlpacaEval ladder returns, this table already says a single-variable
opponent account is incomplete.

**And the confound the registration named is present in the data, measurably.** The two completion
workloads have the two weakest opponents (`0.4865`, `0.5735`) and the two instruction workloads the
two strongest (`0.6594`, `0.6792`); comprehension sits between. Strength and task type move
together across these five arms, so **no reanalysis of them can separate the two**, which is why
this half of feat-178 is registered as suggestive and the intervention is registered as the test.

**One scale worth having beside the spread.** feat-175 measured the same opponent at `0.555` on our
corpus's committed `500`-prompt judged pass; this table reads `0.5735` for the same opponent on the
`850`-prompt binding pass. That `0.018` is what changing the prompt SUBSET of one workload does,
against a between-workload spread of `0.19` --- an order of magnitude larger. The quantity is
therefore about the workload and not about which of its prompts were drawn.

Reproduction: `.venv/bin/python analysis/opponent_by_workload.py --out results`
-> `results/opponent_by_workload.csv`.

### AMENDMENT: L5, the decomposition --- registered 2026-09-22 23:58, before it is computed

P1 is scored above and the ladder is still generating. This adds one analysis to the P1 table and
is committed **before the numbers exist**, in its own section, so that it is auditable as an
addition rather than presented as part of the original design.

**Why it is not a fishing expedition.** The appendix already carries the claim this tests, written
when the third workload landed and repeated when the fourth did: *"the meter is what the workload
changes"* --- over the binding-budget passes `g_sel` spans `0.056` and `g_met` spans `0.092`, and it
is the meter that crosses. feat-174 repeated it a fourth time. If opponent strength is the variable
behind the workload split, it should therefore act **through the meter**, and that is a prediction
the existing sentence makes about numbers nobody has correlated.

**L5.** Over the same arms and the same binding budgets, compute Spearman between opponent strength
and `g_sel` (`D1`) and between opponent strength and `g_met` (`D2`), from the committed h2h CSVs.

| reading | band |
|---|---|
| **THE METER AGAIN** | `abs(rho_met) > abs(rho_sel) + 0.2` |
| **SELECTION, NOT THE METER** | `abs(rho_sel) > abs(rho_met) + 0.2` |
| **UNRESOLVED** | within `0.2` of each other |

The `0.2` margin is there because at `n=5` a Spearman can only take a handful of values
(`+-1.0, +-0.9, +-0.8, +-0.7, ...`), so a bare inequality would resolve on a single rank swap.

**We predict THE METER AGAIN.** A SELECTION, NOT THE METER reading contradicts a sentence the
appendix has carried for three workloads and would require revisiting it rather than being
reported as a curiosity; that is the cost this prediction is here to fix in advance.

**What may not be claimed from L5**: nothing causal, and no significance --- the same `n=5`
permutation floor applies, and the two correlations are computed on the same five arms and are not
independent of each other or of L3.

### L5 SCORED, 2026-09-23 00:05 --- THE METER AGAIN, at the permutation floor

| workload | opponent strength | `g_sel` | `g_met` | `D3` |
|---|---|---|---|---|
| Gutenberg | `0.4865` | `+0.0805` | `-0.0185` | `+0.0990` |
| CoTaEval-QA | `0.5685` | `+0.0235` | `+0.0165` | `+0.0070` |
| ours | `0.5735` | `+0.1218` | `+0.0253` | `+0.0965` |
| MT-Bench | `0.6594` | `+0.0656` | `+0.0906` | `-0.0250` |
| AlpacaEval | `0.6792` | `+0.0835` | `+0.1174` | `-0.0339` |

**`rho(strength, g_met) = +1.000`, exact `p = 0.0167` --- the smallest value `n=5` can produce.
`rho(strength, g_sel) = +0.300`, exact `p = 0.6833`. L5 reads THE METER AGAIN**, and the margin is
`0.7`, not the `0.2` the band asked for.

The meter's gain over the anchor is a **perfectly monotone** function of the opponent's strength
across five workloads, and selection's gain is not a function of it at all. That is the whole
workload split in one line: `D3 = g_sel - g_met`, `g_sel` is roughly flat, `g_met` climbs, so `D3`
falls and eventually changes sign.

**The arithmetic objection is refuted by the analysis's own control, which is why L5 was registered
as a PAIR.** Both gains are differences against the same anchor level, `u_anchor_k0`, so both carry
exactly the same mechanical dependence on it. If a strong opponent forced a large `g_met` by
arithmetic it would force a large `g_sel` too. It does not: `+1.000` against `+0.300` on the same
five numbers. No post-hoc normalisation is needed to say this, and none was computed.

**What the mechanism appears to be, stated as an interpretation and not a measurement.** The
opponent in all five arms is the unconstrained risky model, so "opponent strength" is how much
better the risky model is than the anchor on that workload. A metered decoder is that risky model
pulled toward the anchor, so it inherits the advantage wherever there is one; selection can only
reorder the anchor's own samples and cannot exceed what the anchor could have produced. **This is
the support ceiling in its correct form** --- the appendix retracted that story because selection's
gain barely moved between two workloads while the meter's rose, and that observation is exactly
this one, now generalised to five workloads and with the variable named and measured.

**Three limitations, none of which the numbers above remove.**
`n=5`, so `p = 0.0167` is the floor rather than evidence, and the registration's ban on
significance claims stands. MT-Bench is `80` prompts against the others' `500`--`850`, so one of
the five ranks is far noisier than the rest. And this is a reanalysis of arms run for other
reasons: strength and task type move together across them (P1's own concession), so it identifies
the variable that ORDERS the outcome, not the variable that CAUSES it. The within-workload test is
the AlpacaEval ladder, which was generating while this was scored.

**And the CoTaEval-QA inversion is now located.** Its `D3` is small not because its meter gains
unusually much --- `g_met = +0.0165` sits exactly where its strength predicts --- but because its
`g_sel` is the lowest of the five, `+0.0235` against a median of `+0.0805`. So the one arm that
breaks the opponent account breaks it on the SELECTION side, which is the side the account says
nothing about.

### The sixth arm enters, 2026-09-23 --- L3 strengthens, L4 falls to UNRESOLVED, L5 holds

The registration said unseenbooks enters P1 "when it lands". It landed; the table was re-run with no
other change.

| workload | opponent strength | `g_sel` | `g_met` | `D3` | `D3`/headroom |
|---|---|---|---|---|---|
| unseenbooks | `0.4755` | `+0.0640` | `-0.0400` | `+0.1040` | `+0.2187` |
| Gutenberg | `0.4865` | `+0.0805` | `-0.0185` | `+0.0990` | `+0.2035` |
| CoTaEval-QA | `0.5685` | `+0.0235` | `+0.0165` | `+0.0070` | `+0.0162` |
| ours | `0.5735` | `+0.1218` | `+0.0253` | `+0.0965` | `+0.2263` |
| MT-Bench | `0.6594` | `+0.0656` | `+0.0906` | `-0.0250` | `-0.0734` |
| AlpacaEval | `0.6792` | `+0.0835` | `+0.1174` | `-0.0339` | `-0.1057` |

- **L3: `rho = -0.943`, exact `p = 0.0167` over `720` permutations --- CONSISTENT**, stronger
  than at five.
- **L4: `rho = -0.657`, `p = 0.1750` --- UNRESOLVED.** At five arms it read `-0.700`, exactly on
  the threshold, and this log called it CONSISTENT. A reading on its threshold is the marginal kind
  caution (ap) says will not hold, and the one registered addition moved it.
- **L5: `rho(strength, g_met) = +1.000`, `p = 0.0028` (the `n=6` floor); `rho(strength, g_sel) =
  +0.486`, `p = 0.3556` --- THE METER AGAIN**, with the margin unchanged in kind.

**The registered consequence of L4 is applied.** L4 said: "if [the ordering] does not [survive the
normalisation], say so and withdraw the unification claim." It does not clearly survive, so **the
claim that opponent strength orders `D3` beyond the arithmetic of a difference of win rates is
withdrawn.** What stands is L5, which is a different statement and was registered as the control
for exactly this objection: across six workloads the *meter's* gain rises in perfect rank order with
the opponent's strength while selection's does not, and both gains carry the same arithmetic
dependence on `u_anchor_k0`. So the variation is in the meter and it tracks the risky model's
advantage over the anchor; whether that also orders the *difference* once headroom is taken out is
now UNRESOLVED, and the AlpacaEval ladder --- still generating --- is the test that can settle it
within one workload.

The unseen books have the **weakest** opponent of all six (`0.4755`): the instruction-tuned opponent
is least good at continuing a novel it has not seen, which is the direction the account predicts,
and it is reported as a sixth point, not as confirmation.
