# Session handoff — 2026-09-15 (afternoon, feat-120 IN FLIGHT)

## Current objective

**feat-120 is in flight.** `results/onset_prediction_bookmia.md` is committed and **unscored** —
that is the one unscored pre-registration of the forty-seven in `results/`, and this line is what
`tests/test_preregistration_count.py` checks for. The other forty-six are scored and the manuscript
compiles clean from a deleted PDF.

### feat-120 — the onset split on a third protected corpus (RUNNING)

Three memorisers are fine-tuning, launched 14:18 on GPUs 2 and 4 by
`scripts/run_bookmia_memorisers.sh` (one queue shell per card, no PID capture — caution (x)):

```
queue A, GPU 2, output/logs/bookmia_mem_a.log   kl3m-002-520m, then Pleias-1_2b
queue B, GPU 4, output/logs/bookmia_mem_b.log   phi35mini
```

Expect ~2.5 h per queue: BookMIA's `max-len auto` lands at ~1185 tokens against Gutenberg's 395–669,
so epochs cost 179 s (KL3M) and 219 s (Phi) against 194 s and 81 s on Gutenberg. Padding is dynamic
per batch (`padding=True` at `recipes/finetune_memorizing.py:124`), so the long tail is not paid on
every batch. The fine-tune script prints its own sampled-recall entry-gate verdict when it finishes
(`sampled nv-recall … -> ADMISSIBLE (needs >= 0.10)`), which is caution (a)'s gate, not a greedy one.

**The order from here is the whole point and `scripts/add_pair.sh` deliberately refuses to run the
sweep for this reason:**

1. memorisers finish
2. `analysis/onset_theory.py --pairs-file results/onset_theory_pairs_bookmia.tsv --corpus-file
   data/bench/bookmia100_onset100.jsonl --limit 100 --tag _bookmia` → the parameter-free P1
   predictions, two teacher-forced forward passes per passage, no decoding
3. **append those predictions to the pre-registration and COMMIT** — before the sweep exists
4. `analysis/composition_attack.py` per pair on the committed grid
   `-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2`, `--corpus-file
   data/bench/bookmia100_onset100.jsonl --modes single --limit 100`, `--out output/phase5/fineb_<pair>`
5. `analysis/onset_gutenberg.py --corpus bookmia` and `analysis/onset_ci.py` per pair

**`nvidia-smi` is dead this session and the substitute check matters.** Every call returns
`Failed to initialize NVML: Driver/library version mismatch (NVML library version: 580.173)` — the
host driver was updated under running jobs. Torch is unaffected (it logs one `Can't initialize NVML`
warning and works), so the occupancy check AGENTS.md requires before taking a card has to go through
the CUDA runtime instead:

```
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python -c "
import torch
for i in range(torch.cuda.device_count()):
    free, tot = torch.cuda.mem_get_info(i); print(i, torch.cuda.get_device_name(i), free/2**30)"
```

Run at 14:35 it reads **gpu 0 free 16.1 GiB, gpu 1 free 5.9 GiB** — roughly 63 and 73 GiB held by
**another user** — against gpu 2 free 45.0 and gpu 4 free 50.8, which are ours. So 0 and 1 are NOT
available and the sweeps queue on 2 and 4, the same two cards. This is why feat-120 cannot be
compressed by fanning out.

**Measured timeline** (epoch costs from the live logs, Gutenberg twins for the epoch counts):
KL3M 177 s/epoch and tracking its Gutenberg twin's loss curve, which stopped at epoch ~10, so
queue A frees around 14:50 and starts Pleias (~150 s/epoch × 40 ≈ 16:30). Phi at 218 s/epoch never
hit `--stop-loss` on Gutenberg and should run all 40 → ~16:45, which is the critical path. P1 is
minutes (forward passes only). The three sweeps cost what the Gutenberg ones did, 8,000–14,000 s
each, two cards, so scoring lands late evening. Total arm ≈ 15 GPU-h, inside the 24 that needs asking.

The grid was committed at 14:27 while the memorisers were in epoch 1 and no BookMIA `s_s` existed;
it is the Gutenberg grid verbatim, which is the strongest available evidence it was not shaped to
bracket this corpus.

**Band 3 is the new one and the reason the arm earns its GPU-hours.** Three readings of the same
three pairs give each pair a corpus-to-corpus *range*; the claim is that it sits below that pair's
**CopyBench** bootstrap width (0.2273 / 0.1709 / 0.2866, published in `results/onset_ci.csv` long
before this run). The two-corpus range so far is 0.0488 / 0.0163 / 0.0228 — 3.5× to 10× inside the
yardstick, so the band can fail and is worth stating. Bands 1 and 2 are the Gutenberg arm's verbatim,
because a shared band is the only thing that makes three readings comparable.

**BookMIA's `seen`/`unseen` label plays no role.** Every pair is self-paired — a clean anchor against
a LoRA copy of itself fine-tuned on exactly these passages — so the confound AGENTS.md records
(the halves are not a matched pair under an anchor that saw neither) cannot enter. Said in the
pre-registration too, because a reader will assume otherwise.

The session did eight things, in this order: **reframed the paper to lead with its contribution
(v8)**, **answered a referee report by measurement (v9/v10)**, **read the rendered PDF end to end**,
**ran the paper's own escape hatch from the compute concession and reported that it failed
(feat-116)**, **found and fixed an 8.8% error in the cost model**, **ran the follow-up that
retracted feat-116's headline observation and replaced it with a better one (feat-117)**, and
**took that replacement off the judge, where it did not survive either (feat-118)**, and **priced
the instance of the mechanism that has no scorer at all, which turns out to be both the cheapest and
the best where an answer is checkable (feat-119)**.

Three claims were retracted or qualified this session, all ours, all by arms we designed to test
them. That is the through-line and it is worth preserving: the paper's method is now visibly
applied to the paper's own findings, including the one that was about to become its single
deployer-facing recommendation.

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
| tests | **465 passed**, `./init.sh` exit 0 |
| pre-registrations | **46**, all scored |
| numeric audit | 3,060 literals, 1 expected miss (`64256`, the Comma-7B padded embedding count) |
| compute | 222.9 measured over 234 jobs, "approximately 223" disclosed |
| artifact | 853 files, `MANIFEST.sha256` verified |
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

## Recommended next step

**BookMIA-50 onset** (~11 GPU-h for 3 pairs) is the largest unstarted item and is moderate value now
that onset has nine pairs and two corpora. Everything cheaper that was worth doing has been done.

Two things are on record as **impossible** rather than unstarted, and a future session should not
rediscover them. A judge-free head-to-head against the *metered* decoder cannot be run
(`results/onset_prediction_verifiable.md`): TinyComma is the only openly licensed anchor sharing the
Llama-3 tokenizer and it scores `0.04` on GSM8K. And the anchor-scale axis is capped by the
**licensing frontier**, not by compute — Comma-7B is the largest openly licensed base model we can
obtain, because open-*data* families contain books and would violate the premise the certificate is
written against.

If a session wants a cheap sharpening rather than a new axis: feat-119's table has only the `7.6`B
reward beside majority vote, because that is what the appendix needed. The `0.5`/`1.5`/`3`B caches
exist (`selection_verifiable_rewards_*`), so the full four-scorer cost–accuracy frontier is a
one-line change to `analysis/scorer_free_cost.py` and no GPU. It would not change any claim — every
reward cell already loses to every majority-vote cell — which is exactly why it was left out.

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
