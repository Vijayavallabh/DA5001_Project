# Session handoff — 2026-09-16 (early hours; the SEED arm is RUNNING)

## Current objective

**The seed arm is running.** `results/onset_prediction_seedspread.md` is committed and
**unscored** — the one unscored pre-registration of the forty-nine in `results/`, which is what
`tests/test_preregistration_count.py` checks for. The other forty-eight are scored and the
manuscript compiles clean at 9 of 9 body pages.

### The seed arm — is the onset ratio reproducible under the paper's own recipe? (RUNNING)

The control feat-121 forbade itself from adding after the fact. Same pair, same corpus, same anchor,
same 16-point grid, same 100 swept passages; `--epochs 40` fixed — the recipe **every published
memoriser in this paper uses** — varying `--seed` alone (1, 2, 3), with feat-120's `seed 0` run as
the fourth point. `finetune_memorizing.py` seeds both the per-epoch data shuffle and the LoRA init,
and **every memoriser on record, including all nine pairs of the Section 4 table, was trained at
seed 0**, so the paper has never measured this.

`scripts/run_strength_ladder.sh seeds 1 2 3`, one queue shell, **GPU 2 only**, log
`output/logs/seed_arm.log`. Roughly 6 GPU-h: 3 × (40 epochs at ~100 s, then a ~50-minute sweep).

The script now takes an **axis**: `epochs <n>...` pins seed 0, `seeds <n>...` pins 40 epochs. Two
orthogonal ladders sharing feat-120's run as their corner, which is what makes their spans
comparable. `epochs 10 20 30` still writes the exact directories feat-121 scored.

**Bands**, read against feat-121's measured epoch-only span of `0.4721` on the identical cell:
seed-only span `>= 0.20` → it is run-to-run variation, and the nine-pair table's `0.2874` of
structure sits inside the noise of one pair re-trained; `< 0.10` → it is strength, and the table is
confounded by a measurable variable; otherwise inconclusive. Committed secondary, so it cannot be
post hoc: pooling both ladders gives **seven** points on one pair, where the exact-`p` floor is
`1/2520` and this question can reach significance for the first time.

Score with `analysis/strength_ladder.py` once extended to the seed axis (it currently hard-codes
the epoch POINTS list — extend it, do not duplicate it).

The session ran ten arms. The last two are the ones a fresh reader should start with, because
together they removed a claim the paper had built a section around:

- **feat-120** took the onset result to a **third** protected corpus. Two of three committed bands
  failed; the ordering inverted; the sentence "leakage beginning after the certificate has gone
  vacuous is a property of the pair" was retracted.
- **feat-121** asked the question feat-120 could not answer — corpus or memoriser? — and found that
  **one pair, held at one corpus and one anchor, moves its onset ratio 1.64× more than the entire
  nine-pair table moves across all nine pairs**, purely by changing how long its memoriser trained.

**Five claims were retracted or qualified this session, all ours, all by arms we built to test
them.** That is the through-line worth preserving.

### feat-121 — corpus or memoriser? **INCONCLUSIVE on the committed metric, and the measurement is the finding**

Pleias-1.2B on BookMIA, the exact cell that inverted, with three further memorisers differing in
`--epochs` alone (10, 20, 30) and feat-120's own 40-epoch run as the fourth point. Everything else
identical; strength **measured** by each point's own sampled `k=-1` arm, never read off the knob.

```
point        sampled k=-1   onset    ratio   95% CI            no-x
epochs=10       0.5470      2.669   0.8756  [0.7937, 1.2303]   0.0%
epochs=20       0.2269      4.108   1.3477  [0.9667, 1.7475]   0.0%
epochs=30       0.9149      2.923   0.9590  [0.6827, 0.9819]   0.0%
epochs=40       0.1504      4.006   1.3142  [1.0102, 1.5941]   0.4%
```

- **The arm is valid**: the knob produced a `6.08x` strength span against a committed `3x`.
- **Committed verdict INCONCLUSIVE**: band 1 needed `rho <= -0.8` **and** span `>= 0.15`. The span
  fired at `0.4721`; `rho` did not, at `-0.600` (exact `p = 0.4167`, floor `0.083`).
- **The knob is not monotone and the design never assumed it was** — 20 epochs gave a *weaker*
  memoriser than 10, which is why every band correlates against measured strength.
- **What needs no band**: a spread of `0.4721` within one pair against the nine-pair table's whole
  `0.2874`. It reproduces both ends of that range alone, and the property the tokenizer split turns
  on flips sign inside the pair — `[0.68, 0.98]` at the strongest memoriser, `[1.01, 1.59]` at the
  weakest, non-overlapping. (The extremes comparison is POST HOC; the spread was committed.)

**Not resolved, and stated as plainly as the finding:** strength versus run-to-run variation of the
fine-tune. Four epoch counts are four optimisation trajectories, and the control — several seeds at
one epoch count — is forbidden by the pre-registration after the fact. **Both readings cost the
same**: if strength, the nine-pair table is confounded by a variable it never controlled (its own
memorisers span sampled `k=-1` from `0.2696` to `0.9091`); if noise, the ratio carries a within-pair
uncertainty near `0.47` that no bootstrap interval reports and the `0.2874` spread sits inside it.

**As pre-committed, no outcome restored the retracted sentence**, and none did.

### A rendering defect no build check can see, found while paying the page budget

A `\label` attached to a `\paragraph` captures no counter, so `\ref` resolves it to the enclosing
`\section`. Two such refs side by side therefore rendered **in the compiled PDF** as
`Appendices I, I` and `Appendices I--I`. tectonic exits 0, the overfull count is 0, `??` is 0, and
the page shows a real appendix letter — just the same one twice. Same class as cautions (y) and (z).
`tests/test_reference_targets.py` (2) catches it, and was **demonstrated to fail** on the
reintroduced defect rather than assumed to work. `sec:onset` (23 refs) and `app:bookmia` are the
same construction and are *correct*, because their enclosing section is the right target.

### Compute is now disclosed as an UPPER bound, and why

`compute_hours.py` bills a log from birth to last write. This session's **gated queue shells poll
while holding no card** — `bookmia_p1` ~2.63 h, `bookmia_sweep_gpu2` ~2.53 h, `bookmia_sweep_gpu4`
~2.53 h — so roughly **7.7 of the 247.6 GPU-h is a sleeping shell**. The LLM Usage statement
therefore says "at most $248$ GPU-hours" rather than "approximately". This is a new over-billing
mode introduced by this session's gate loops; a future session adding a waiter should either trace
its sleeps or expect the same inflation.

### feat-120 — the onset split on a third protected corpus: **TWO OF THREE BANDS FAIL**

**Complete and scored.** All forty-seven pre-registrations are scored; nothing is running; no GPU
work is outstanding. This arm refuted a claim the paper stated as established, and the manuscript
now says so in the three places that made it.

**The design.** Same three anchors that already carried a CopyBench reading and a Gutenberg reading
— KL3M-520M, Pleias-1.2B, Phi-3.5-mini — given a third memoriser on 600 BookMIA passages
**stratified round-robin over all 31 books**, swept on a 100-passage prefix of that set. Everything
held fixed; only the protected work changed, for the second time. Bands 1 and 2 are the Gutenberg
arm's verbatim; band 3 is new and is what three readings buy that two cannot.

```
pair            k=-1    k=0    onset   ratio  95% CI           no-x   pred/meas   Gutenberg  novels
KL3M-520M      0.7326  0.000   2.465  1.0138  [0.970, 1.131]   0.0%     0.891       1.102     1.053
Pleias-1.2B    0.1504  0.000   4.006  1.3142  [1.010, 1.594]   0.4%     0.652       0.895     0.878
Phi-3.5-mini   0.4425  0.000   2.607  0.9920  [0.867, 1.370]   0.0%     1.004       0.949     0.926
```

- **Band 1 FAILS.** Pleias' `0.652` is outside the committed `[0.7, 1.4]`. Eq. (req)'s *level*,
  which transferred to Gutenberg, does not transfer twice.
- **Band 2 INVERTS — the strongest committed negative, and it fires twice over.** The coarse
  Pleias pair (`1.3142`) is now *above* the fine KL3M pair (`1.0138`); and KL3M's interval
  `[0.970, 1.131]` no longer excludes 1, where CopyBench `[1.0163, 1.2436]` and Gutenberg
  `[1.0744, 1.4227]` both did. That exclusion was the whole basis for "leakage begins after the
  certificate has gone vacuous", so the sentence is retracted.
- **Band 3, one pair exceeds.** KL3M and Phi vary across three corpora by `0.0882` and `0.0659`,
  which is `2.6x` and `4.3x` *inside* their own CopyBench bootstrap widths (`0.2273`, `0.2866`);
  Pleias varies by `0.4358` against `0.1709`. Corpus-sensitive for one pair, named.

**The grid extension confirmed rather than rescued, and that is the methodological point.** Pleias'
no-crossing fraction read **43.1%** against `0.0%` and `0.0%`, on a grid whose ceiling its upper
bound sat `0.056` below — caution (g) exactly. The extension was licensed by `appendix_seed.tex`'s
rule, cited in the pre-registration at 14:27, with grid `{4.6, 5.3, 6.6}` taken mechanically from
the KL3M-1.7B precedent's multipliers and committed *before* the run. Result: no-crossing
`43.1% -> 0.4%`, upper end `1.360 -> 1.594`, onset **unmoved at 4.0058 to four decimals**. The
reading was never a ceiling artefact; the grid was too short to prove it. Both grids are reported
(`results/onset_bookmia_committed_grid.csv` holds the unextended scoring).

**The confound is measured and does NOT rescue anything.** Pleias' memoriser is much the weakest,
and its sampled `k=-1` falls `0.909 -> 0.517 -> 0.150` across the three corpora while its ratio
rises `0.878 -> 0.895 -> 1.314`. But over all nine (pair, corpus) cells the rank correlation is
`rho = -0.317` at exact `p = 0.4101` — **post hoc, no band, not significant**. A named caveat for
one pair, not a general account, and never used to set aside a band that fired. Pleias' `0.1504`
clears the committed `0.10` gate and it enters; the gate was not raised after the fact. Because a
memoriser is fine-tuned on the corpus it is measured against, this design cannot separate corpus
from memoriser — a limitation of the design, not a defence of the claim.

**What survives, and it is the paper's actual claim.** Every onset on every pair and every corpus
still lands within about `1.3x` of `s(x)`, so `prop:threshold` is not bookkeeping. What died is the
finer structure — which pairs sit above 1 and which below — and the rule now stated in the paper is
that an onset ratio transfers across corpora to its order of magnitude and no more finely.

**Manuscript changes.** `appendix_robustness.tex`'s subsection is retitled "A second corpus, then a
third" and the retracted sentence is kept *visible as a retraction* beside the evidence that
refutes it, so a reviewer who read an earlier draft or the pre-registration finds the withdrawal in
the same place. `appendix_limitations.tex` carries the same retraction. `onset.tex` (body) gains
"though on two further protected corpora it reaches 1.31 and which pairs exceed 1 changes, so only
the level transfers". The abstract gains ", and to 1.31 on two further corpora".

**Two defects of mine this arm, both caught by tests rather than by reading.**
1. I appended the P1 block with `partition("## Scoring log")` instead of the newline-prefixed form,
   so it matched the backticked mention in the file's own third sentence and inserted the block
   mid-sentence — **caution (r) verbatim**, in the one file whose entire purpose is ordering.
   Repaired; content and commit times unaffected, only placement.
2. My first abstract rewrite replaced "nine pairs with nine distinct anchors" with "nine anchors and
   three protected corpora", silently dropping the pair count.
   `test_abstract_consistency.py::test_the_abstract_the_intro_and_the_onset_section_agree_on_the_pair_count`
   caught it, which is exactly the drift that test exists for.

Earlier in the arm I also forecast the fine-tune stop-loss twice off three-epoch stretches and was
wrong in both directions (KL3M read 0.0281 at epoch 16, 0.1439 at 21, then stopped at 0.0169 at 26).
**A LoRA at `lr 3e-4 rank 128` oscillates hard near convergence: read `recipe.json`'s `final_loss`,
never the curve.** Only the per-epoch cost is worth quoting as a forecast.

**ONE GPU AT A TIME, from 21:00 (user instruction, now in AGENTS.md above the Compute block).** All
future processes share **GPU 2**. This supersedes the Working Rules allowance that "GPU arms may
queue in parallel when separate cards are free" — arms may still queue, but on one card, with
caution (x)'s one-queue-shell repair applying within it. The instruction allowed this arm's two
in-flight sweeps to finish on 2 and 4; nothing was killed and nothing was queued behind them.

**`nvidia-smi` is dead on this box** — the host driver was updated under running jobs and every call
returns `Failed to initialize NVML: Driver/library version mismatch (580.173)`. Torch is unaffected,
so the occupancy check AGENTS.md requires goes through `torch.cuda.mem_get_info` instead. Run at
14:35 it showed **GPUs 0 and 1 held by another user** (16.1 and 5.9 GiB free), which is why this arm
ran on 2 and 4 and could not be compressed.

### v8 / v9 / v10 / the first read-through — history, carried for context

v7 argued the dichotomy and reached selection anchoring on page 5. v8 swapped the roles: §2 is
selection anchoring, §3 its experiments, §4 the dichotomy, §5 the deployed instance. Title is *Two
Nats, Not Two Thousand: Spending the Copyright Budget on the Draw Instead of the Token*. v9/v10
answered a 5/10-reject referee report on four counts by measurement — the composition order (192
pathwise / 332 KL / 2 metered realised / 0 metered certified), feat-113's order-averaged
head-to-head (**REVERSAL CONFIRMED**, `+0.0645 [+0.030,+0.0995]`, with my pre-registered prediction
wrong), feat-114's serving cost, and feat-115's anchor vetting over eighteen models. The
read-through then caught two estimand defects the fixes themselves had introduced. Every section
from v7 is kept beside its replacement as `*_v7_2026-09-14.tex`.

### feat-116 — the 0.5B scorer does not carry the gain

`serving_cost.csv` carried a row marked `0.5B, NOT RUN`: the arithmetic saying that because the
reward model dominates the price, a small scorer would cut the cost several-fold. An unmeasured
counterfactual is the thing this paper exists to object to, so it was pre-registered (`1cc982d`,
five bands) and run. `Qwen2.5-0.5B-Instruct`, identical template and reward, re-scoring the same
32,000 cached candidates, both scorers judged over the whole nested grid under feat-113's corrected
protocol — 8,260 calls, `0.23` GPU-h.

**The route fails and I was wrong on three of four bands.** F1 `FAILS` (`+0.0220 [+0.0000,+0.0440]`
at `n=64` against a committed `WORKS` in `[0.03,0.09]`), F2 `COSTLY` at `-0.0855`, F4
`MATCHED-COMPUTE LOSS` at `-0.0395` against a committed `PARITY`, F3 `NO CROSSING`. F3 keeps its
margin rather than its label in the prose: at `n=16` the small scorer reaches `+0.0395` against the
meter's `+0.0400`, five ten-thousandths short with the intervals overlapping, so the honest reading
is *indistinguishable from*, not below.

### The cost model priced its models by their names

Found while measuring the scorers for feat-117. Counted off the loaded checkpoints,
`Qwen2.5-7B-Instruct` holds **`7.6156`B**, not `7.0`; TinyComma `1.7586`B; Llama-3.1-8B-Instruct
`8.0303`B. The scorer term is the one selection pays `n` times, so every serving-cost number was
understated by `8.8%`: **`7.2x` is `7.7x` and `57.5x` is `61.3x`**. Corrected everywhere, including
the appendix table's twelve cells. feat-116's own docstring warned about this exact class of error
and I made it anyway on the three constants I did not re-measure. `serving_cost.py` now carries
measured counts with the reproduction one-liner, and `test_compute_matched.py` fails if any constant
is ever equal to the round number in its model's name.

### feat-117 — NO TURNOVER FOUND, and what replaced it

feat-116 reported, unregistered, that the 0.5B curve "peaks at `n=16` and falls" and drew from it
that `log n` is not a free knob. feat-117 put two scales between `0.5`B and `7.6`B
(`Qwen2.5-1.5B`, `Qwen2.5-3B`), registered the terminal drop `g(64) − g(16)` as the test, and judged
all four scorers **in one pass** — 10,722 calls, 4,361 distinct served completions, `0.53` GPU-h
measured. One pass is not a convenience: the comparison is between scorers, and this paper's own
instrument checks forbid quoting judged levels across passes.

**G0 replicates to `0.001`** on all three reference arms against a gate that allowed `0.04`. That is
the strongest evidence this paper has that its judged *gains* — taken over a shared control with
position removed by construction — are stable even though its judged *levels* are not.

**G3 = NO TURNOVER FOUND**, against my committed `BOUNDARY AT 1.5B`. The 0.5B terminal drop is
`-0.0160 [-0.0340, +0.0010]`: negative, interval containing zero by a thousandth, so `FLAT`. The
committed consequence was a retraction and it is executed — the appendix says we made the reading,
registered it and lost it, and the `log n`-is-not-a-free-knob claim is out of the paper and out of
this file. What survives as *description*, not as the registered test, is the shape contrast:
Spearman(gain, log n) is `0.5429` at `0.5`B and exactly `1.0` at each of `1.5`B, `3`B, `7.6`B.

**The finding it was not built for is larger: capability saturates early --- on the judged
axis. feat-118 below took this off the judge and it did not hold.** Of the three adjacent
steps at `n=64` only the first separates — `1.5`B over `0.5`B `+0.0700 [+0.0500,+0.0900]`, `3`B over
`1.5`B `+0.0040 [-0.0145,+0.0225]`, `7.6`B over `3`B `+0.0095 [-0.0090,+0.0285]`. A `1.5`B scorer
reaches **`87.3%`** of the `7.6`B gain for **`35.2%`** of the cost (`21.59x` against `61.29x`), so
**the `61.3x` the paper concedes is the price of the scorer we happened to use, not of the
mechanism** — which retracts the other half of feat-116's Limitations sentence, that the cost was
"intrinsic at the scales we tested". The cost column carries no committed band and is labelled post
hoc; the crossing rule is feat-116's, applied unchanged.

**One instability recorded rather than hidden.** A crossing *cell* is a thresholded statistic and
inherits none of the stability the gains showed: the `7.6`B crossing sits at `n=8`/`7.66x` in one
pass and `n=16`/`15.32x` in the other, because `sel7b_n8` read `+0.0415` then `+0.0360` — `0.0055`
apart, far inside the judge's floor, but straddling the meter's `+0.0400`. The paper quotes the
`n=64` column, which moved by `0.001`.

Without a judge, `scorer_scale_agreement.csv` (post hoc, no band) gives the same ordering in reward
space: Spearman against the `7.6`B reference is `0.1333`, `0.4185`, `0.4926` at `0.5`/`1.5`/`3`B and
the same-draw rate at `n=64` is `0.052`, `0.132`, `0.228` against `0.016` by chance. Large step at
the bottom, small ones above it.

### feat-118 — and the replacement does not survive the judge-free axis either

feat-117's saturation result was about to become the paper's one practical recommendation — *use a
`1.5`B scorer, keep `87%` of the gain, pay `35%` of the cost* — and every number in it came from an
instrument this paper calls UNUSABLE. GSM8K exact match has no judge. Same `500 × 64` cached
Comma-7B candidates, all four rewards, bands committed at `8b97871` before the run, with TriviaQA
committed alongside so it could not become a post-hoc rescue.

**H0 `MATCHES`**, plus a free check: majority vote touches no reward model, so all four per-scorer
runs must reproduce it exactly — and do, which pins the cached-generation path.

**H3 = `DISAGREES, OTHER` on GSM8K and `DISAGREES, NO SCORER EFFECT` on TriviaQA, against my
committed `AGREES` on both.** Scorer scale does matter without a judge (`7.6`B over `0.5`B at
`n=64` is `+0.0700 [+0.0300,+0.1120]`) but **not where the judge said**: the first step, `1.5`B over
`0.5`B, is `-0.0060 [-0.0500,+0.0360]` against the judged `+0.0700 [+0.0500,+0.0900]`, no adjacent
step resolves alone, and the largest scorer is still the best. On TriviaQA no scale helps and every
reward arm *declines* in `n`.

**Both axes agree scorer scale is worth seven to eight points end to end** — `+0.0835` judged,
`+0.0700` judge-free — **and disagree completely about where it is bought**: all in the first step
with a judge, gradually and still rising at `7.6`B without one.

Consequence applied as committed: the `87%`/`35%` sentence is **qualified to the judged workload**
in Section 2 and Limitations rather than withdrawn, since GSM8K's H1 separates. The test that
pinned it now forbids the unqualified form and requires the disagreement beside it. New appendix
`app:judgefreescale`.

Stronger for it: the paper's central claim is now measured **without** an instrument — `+0.0700` of
exact match turns on the scorer alone with `log n` identical throughout — and majority vote, no
reward model and the same certificate, reaches `0.546` where the best reward reaches `0.386`.

### feat-119 — the cheapest instance is also the best one

Post hoc and labelled so: no band, no new run, a join over quantities already measured, and not
named `onset_prediction_*` so it cannot inflate the count (feat-115's rule).

feat-114/116/117 established that selection's cost is the **reward model**, not the anchor.
**Majority vote has no reward model** — self-consistency serves the modal answer over the same `n`
anchor draws, so its score is a regex and costs no forward pass, and it carries the identical
`log n` certificate because Proposition 1 assumes nothing about the score and a mode is a score.

| rule | `n` | cost | GSM8K | TriviaQA |
|---|---|---|---|---|
| majority vote | 8 | **`1.44x`** | `0.466` | `0.322` |
| reward `7.6`B | 2 | `1.92x` | `0.352` | `0.282` |
| majority vote | 32 | **`5.75x`** | **`0.546`** | `0.328` |
| reward `7.6`B | 64 | `61.29x` | `0.386` | `0.266` |

**Every scorer-free cell beats every reward cell on both tasks** — `3.42x` the gain for `9.4%` of
the cost at `n=32`, and on TriviaQA the reward goes *negative* at every `n ≥ 8`. So `61.3x` is the
price of putting a reward model in the loop, not the price of the mechanism.

Three limits travel with the numbers in the CSV, the note and the appendix, because this is the kind
of figure that gets quoted out of context: post hoc; needs a **canonical answer**, so it does not
transfer to the free-form judged workload; and no metered decoder ran on GSM8K, so the `x metered`
column is our cost denominator and not a measured head-to-head. The appendix also states what it
does **not** say — majority vote reaches `0.546` against the risky model's `0.786` greedy, so this
is a comparison among mechanisms that carry a certificate, not a claim to beat the model one exists
to bound.

New appendix `app:scorerfree`; Section 2 gains a clause and drops the `1.5`B detail that Limitations
and the appendix already carry; the body figure went `0.74 → 0.70\textwidth` to pay for it, rendered
at 190dpi and checked.

### The final read-through

All nine body pages read as rendered. Three fixes, one substantive: **the abstract claimed selection
"does better" at one fifty-fourth of the divergence and never mentioned compute** — which is exactly
the referee's charge that we chose the axis we win on. It now says "at one fifty-fourth of the
divergence **and sixty-one times the forward-pass cost**", paid for by cutting a rhetorical sentence
the introduction argues properly anyway, and the abstract still ends on line 041. Also: the
contributions bullet cited §3 for a ratio only §2 and the appendix carry, and Limitations named two
things as open while calling one "the main open problem".

Checked and **not** defects: the `+0.054` in §2 is Table 1's single-order judge-B `n=8` row and the
placement arms it is compared against share that protocol, 500 prompts, gains over an anchor-alone
control at the same `2.08` pathwise nats (`results/placement.csv`); the `165.0` nats in the
composition paragraph is the `k=3` realised KL, a different arm from the `k=10` workload's `171.3`,
and both are labelled where they appear.

## State

| | |
|---|---|
| manuscript | `~/sub/satml/iclr_2027.tex`, **9 of 9 body pages**, 55 total |
| build | exit 0, **0** overfull, **0** unresolved, **0** literal `**`, page 10 body-free |
| tests | **483 passed**, `./init.sh` exit 0 |
| pre-registrations | **49**; 48 scored, `onset_prediction_seedspread.md` committed-and-running |
| numeric audit | 3,197 literals, 1 expected miss (`64256`, the Comma-7B padded embedding count) |
| compute | **247.6** measured, disclosed as **"at most 248"** (≈7.7 of it is gated shells polling with no card); fine-tune bound **40** |
| artifact | **871** files, `MANIFEST.sha256` verified |
| anonymity | 0 "our earlier audit", 0 affiliation; 4 hits, all `(Vijayavallabh, 2026)` and its bib entry |

## Files changed this session

**Manuscript.** Every body section rewritten or reordered for v8; `selection.tex` and
`iclr_closing.tex` rewritten again for the cost frontier and the saturation result; `iclr_intro.tex`
and the abstract in `iclr_2027.tex` for the compute axis and the reference fix. New appendix
sections `app:frontier`, `app:saturation`, `app:judgefreescale` and `app:scorerfree` in
`appendix_selection.tex`, alongside the three v9/v10 paragraphs (order-averaged head-to-head,
two-order composition, anchor vetting); the `app:saturation` paragraph carries a forward pointer to
the judge-free rebuttal. Section 2's cost sentence was rewritten three times as the cost story
changed and ends up saying the simplest true thing: the price is the scorer's, and the cheapest
scorer is none. The body figure went `0.80 → 0.74 → 0.70\textwidth` paying for those edits, rendered
and read at each step.
`appendix_proofs.tex` (duplicate label removed, threshold proof generalised), `appendix_related.tex`
(Kalai/Chen), `references.bib` (+`panickssery2024llm`).

**Repo.** New analysis scripts `order_averaged_h2h`, `serving_cost`, `anchor_vetting`,
`compute_matched`, `scorer_agreement`, `scorer_scale`, `verifiable_scorer_scale`; new launcher
`scripts/run_verifiable_scorer_scale.sh`. New `results/` CSVs for each of those arms plus the four
reward caches `selection_rewards64_qwen{05b,15b,3b}.csv` and
`selection_verifiable_rewards_{,tqa_}comma7b_qwen{05b,15b,3b}.csv`, and the six per-scorer
`selection_verifiable{,_tqa}_comma7b_qwen*.csv`. Four pre-registrations:
`onset_prediction_{order_averaged_h2h,compute_matched,scorer_scale,verifiable_scorer_scale}.md`.
New `tests/test_{compute_matched,scorer_scale,verifiable_scorer_scale,scorer_free_cost}.py`.

feat-119 added `analysis/scorer_free_cost.py`, `results/scorer_free_cost.csv` and its note; it is a
**post-hoc join with no committed band**, so the note is `scorer_free_cost_note.md` and
deliberately *not* an `onset_prediction_*.md` — the same rule feat-115 followed, so a re-analysis
cannot inflate the pre-registration count. `analysis/serving_cost.py` gained the majority-vote rows
(`n P_anchor` and no scorer term).

Two existing scripts were patched rather than duplicated. `scripts/snapshot_manuscript.sh` follows
nested `\input`. `analysis/selection_verifiable.py` gained `--reward-tag`, which names the reward
cache and output CSV without touching the generation paths, and had a latent defect fixed: its
reward arm's label was the string `pointwise reward (Qwen2.5-7B)` hardcoded, so `--reward-model`
would have silently mislabelled every row and made four scorers indistinguishable in the CSVs. With
defaults it reproduces `selection_verifiable_comma7b.csv` byte for byte.
`tests/test_{selection_claims,imitation_cost}.py` updated.

### feat-121 (this session, COMPLETE)

```
analysis/strength_ladder.py              NEW  correlates ratio vs MEASURED k=-1; refuses < 3 points
scripts/run_strength_ladder.sh           NEW  one queue shell, GPU 2 only, fine-tune then sweep
tests/test_strength_ladder.py            NEW  5 tests
tests/test_reference_targets.py          NEW  2 tests; catches "Appendices I, I"
results/onset_prediction_strength.md     NEW  committed a9360cb, SCORED 02:30
results/strength_ladder.csv              NEW  the four points
~/sub/satml/sections/onset.tex                 body: the level is the claim, the ordering is not
~/sub/satml/sections/appendix_robustness.tex   NEW subsection with the four-point table
~/sub/satml/sections/appendix_limitations.tex  the nine-pair table is evidence about the LEVEL
~/sub/satml/sections/selection.tex             ref fix (one number, pointer preserved)
~/sub/satml/sections/iclr_closing.tex          ref fix; "residual fifth" sentence cut
~/sub/satml/iclr_2027.tex                      abstract range; compute -> "at most 248"
```

### feat-120 (this session, COMPLETE)

```
analysis/build_bookmia_onset_subset.py   NEW  stratified round-robin subset builder
analysis/onset_gutenberg.py              PATCHED  --corpus {gutenberg,bookmia}; default byte-identical
scripts/run_bookmia_memorisers.sh        NEW  queue shell per card, flags verbatim from Gutenberg
scripts/run_bookmia_p1.sh                NEW  waits for all three memorisers, runs P1, STOPS
scripts/run_bookmia_sweeps.sh            NEW  pair table inside the script; git-checked P1 gate
tests/test_bookmia_onset.py              NEW  8 tests
scripts/run_bookmia_sweeps.sh            NEW  GATE-BEGIN/GATE-END, git-checked P1 gate
results/onset_prediction_bookmia.md      NEW  committed 14:27, SCORED 22:05
results/onset_bookmia.csv                NEW  extended-grid scoring (reading of record)
results/onset_bookmia_committed_grid.csv NEW  the unextended scoring, as the rule requires
results/onset_theory_bookmia.csv         NEW  P1, committed dad60ec before any sweep
~/sub/satml/sections/appendix_robustness.tex   RETRACTION + the third corpus
~/sub/satml/sections/appendix_limitations.tex  same retraction
~/sub/satml/sections/onset.tex                 body claim qualified
~/sub/satml/iclr_2027.tex                      abstract range; compute 223 -> 244
results/onset_theory_pairs_bookmia.tsv   NEW  label / memoriser / anchor
data/bench/bookmia100_onset{600,100}.jsonl  NEW, gitignored, rebuildable
```

## Recommended next step

Nothing is in flight. **The highest-value next move is a read-through of Section 4 and its
appendices with feat-121 in hand**, because the section was written when the nine-pair ratios were
read as properties of pairs and that reading is now gone. Concretely:

1. `sections/appendix_onset.tex` still presents the seed-word gradient (`rho = -0.958`, exact
   `p = 0.0002`) as an explanation of the residual spread across nine pairs. It is a correlation
   over **nine differently-trained memorisers**, and feat-121 shows one pair traverses more than
   that whole spread on its own. Limitations now says so; the appendix does not. Either qualify it
   there too or cut the causal reading. The body sentence was already hedged to "part of the
   residual tracks…".
2. `sections/appendix_seed.tex` carries the same ratios in three tables (~19–26, ~250, ~311–313) as
   per-pair properties, and its "adversary holds >10 words / <=10 words" split (`0.878`–`0.926`
   against `0.993`–`1.166`) is exactly the structure feat-121 says is not resolvable at that
   precision. This is the biggest remaining overclaim in the paper.
3. `results/onset_table.csv` and `sections/appendix_onset.tex`'s nine-pair table would be more
   honest with a memoriser-strength column (each pair's sampled `k=-1` is already on disk). Zero
   GPU, and it lets a reader see the confound rather than being told about it.

**If a new arm is wanted**, the one feat-121 explicitly could not run is the clean separation:
several fine-tunes at **one** epoch count under **different seeds**, same pair and corpus. Three or
four seeds would say whether the `0.47` within-pair spread is memoriser strength or run-to-run
variation. feat-121's pre-registration forbids folding that into itself, so it needs its own
pre-registration and its own bands. **One card (GPU 2)**, roughly 4–5 GPU-h.

Two things remain on record as **impossible** rather than unstarted. A judge-free head-to-head
against the *metered* decoder cannot be run (`results/onset_prediction_verifiable.md`): TinyComma is
the only openly licensed anchor sharing the Llama-3 tokenizer and it scores `0.04` on GSM8K. And the
anchor-scale axis is capped by the **licensing frontier**, not by compute — Comma-7B is the largest
openly licensed base model we can obtain, because open-*data* families contain books and would
violate the premise the certificate is written against.

## What this session added to the record

Cautions stand at **twenty-seven**; three were added earlier in the session — **(y)** `**text**` is
markdown and renders as literal asterisks, **(z)** a `\label` in both body and appendix silently
redirects every `\ref`, **(aa)** `served_prompt()` reconstructs the judge's prompt and disagrees
across arms in 455 of 500 cases, with judge C disclosed as the opponent's own checkpoint.

feat-116, feat-117 and feat-118 added no new caution, because every trap they could have hit was
already written down. What they added is the clearest worked example this project has of what the
cautions are *for*, run twice in one session:

- **A shape you can see in a table is not a finding until an interval says so.** feat-116 read a
  curve off a table and drew a deployer-facing warning from it; feat-117 registered the test and
  the warning did not survive.
- **A result measured on one instrument is a claim about that instrument until a second one agrees.**
  feat-117 replaced the withdrawn warning with capability saturation, which was one step from
  becoming the paper's only practical recommendation; feat-118 took it to an axis with no judge and
  the two disagree about where the effect lives. It is qualified in the paper, not quietly dropped,
  and a test forbids the unqualified form.

Both retractions are in the appendix in the paper's own voice. That is the habit worth keeping.

feat-119 added a third, about a number rather than a claim: **a figure that will be quoted out of
context needs its limits attached at every copy.** The scorer-free result is the most quotable thing
in the paper — *the cheapest instance is also the best* — and it is true only where an answer is
canonical, only as a cost model rather than a measured head-to-head on that task, and only among
mechanisms that carry a certificate. Those three limits are in the CSV's own `limits` column, in the
note and in the appendix, and a test fails if any copy loses them.

Three habits, all earned the hard way here:

- **Render the page and look at it.** A figure font change made to buy page space collided the axis
  labels and put the legend on the data; the build was clean and every automated check passed.
- **Run the *full* gate after adding any float.** A new appendix table produced a 52.8pt overfull
  box that the page-budget check does not see.
- **Measure a parameter count, never read it off a model's name.** `Qwen2.5-7B-Instruct` is
  `7.6156`B and pricing it at `7.0` understated a headline number by `8.8%` for two commits.
- **Open the run log before reusing a protocol** (caution (v), which earned its keep again).
  TriviaQA was **5-shot**, not the 8-shot default, and a guessed protocol would have compared the
  new scorers against 7.6B rows produced under different conditions.
