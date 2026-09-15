# Session handoff — 2026-09-15 (midday, after feat-118)

## Current objective

Nothing is running. All forty-six pre-registrations in `results/` are scored and the manuscript
compiles clean from a deleted PDF.

The session did seven things, in this order: **reframed the paper to lead with its contribution
(v8)**, **answered a referee report by measurement (v9/v10)**, **read the rendered PDF end to end**,
**ran the paper's own escape hatch from the compute concession and reported that it failed
(feat-116)**, **found and fixed an 8.8% error in the cost model**, **ran the follow-up that
retracted feat-116's headline observation and replaced it with a better one (feat-117)**, and
**took that replacement off the judge, where it did not survive either (feat-118)**.

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
| manuscript | `~/sub/satml/iclr_2027.tex`, **9 of 9 body pages**, 54 total |
| build | exit 0, **0** overfull, **0** unresolved, **0** literal `**`, page 10 body-free |
| tests | **458 passed**, `./init.sh` exit 0 |
| pre-registrations | **46**, all scored |
| numeric audit | 3,006 literals, 1 expected miss (`64256`, the Comma-7B padded embedding count) |
| compute | 222.9 measured over 234 jobs, "approximately 223" disclosed |
| artifact | 849 files, `MANIFEST.sha256` verified |
| anonymity | 0 "our earlier audit", 0 affiliation; 4 hits, all `(Vijayavallabh, 2026)` and its bib entry |

## Files changed this session

**Manuscript.** Every body section rewritten or reordered for v8; `selection.tex` and
`iclr_closing.tex` rewritten again for the cost frontier and the saturation result; `iclr_intro.tex`
and the abstract in `iclr_2027.tex` for the compute axis and the reference fix. New appendix
sections `app:frontier` and `app:saturation` in `appendix_selection.tex`, alongside the three
v9/v10 paragraphs (order-averaged head-to-head, two-order composition, anchor vetting).
`appendix_proofs.tex` (duplicate label removed, threshold proof generalised), `appendix_related.tex`
(Kalai/Chen), `references.bib` (+`panickssery2024llm`).

**Repo.** New `analysis/{order_averaged_h2h,serving_cost,anchor_vetting,compute_matched,scorer_agreement,scorer_scale}.py`;
new `results/{order_averaged_h2h,serving_cost,anchor_vetting,compute_matched,compute_matched_bands,compute_matched_per_prompt,compute_matched_scorer_agreement,scorer_scale,scorer_scale_bands,scorer_scale_per_prompt,scorer_scale_agreement}.csv`
plus the reward caches `selection_rewards64_qwen{05b,15b,3b}.csv`; pre-registrations
`onset_prediction_{order_averaged_h2h,compute_matched,scorer_scale}.md`; new
`tests/test_{compute_matched,scorer_scale}.py`; `scripts/snapshot_manuscript.sh` follows nested
`\input`; `tests/test_{selection_claims,imitation_cost}.py` updated.

## Recommended next step

**Take the saturation result off the judge.** It is now the session's most useful finding and it
rests entirely on an instrument this paper itself calls UNUSABLE (order consistency `0.24`–`0.35`).
There is a judge-free axis that can test it directly, and everything it needs is already on disk:
`output/phase5/verifiable/anchor_comma7b_n64.jsonl` holds 500 GSM8K problems × 64 anchor
candidates, and `results/selection_verifiable_rewards_comma7b.csv` is the 7B reward over all 32,000
of them. Re-score those same candidates with the `0.5`B, `1.5`B and `3`B rewards and read **exact
match** at each `n` — no judge, no generation, no position bias, roughly `1` GPU-h for three reward
passes with the scoring itself on CPU.

That answers what the judged arm can only gesture at: does capability saturate on a task with a
ground-truth answer, and at the same scale? If it does, the claim that the divergence axis is the
mechanism's and the utility is the scorer's is established without the instrument Section 3
distrusts — and `results/selection_verifiable_comma7b.csv` already holds the `n`-sweep for the 7B
scorer to compare against. It needs its own pre-registration with a band on the saturation step, and
it should commit in advance to reporting a *non*-saturating judge-free curve as evidence against the
judged result rather than as a separate phenomenon.

Second, unchanged: **BookMIA-50 onset** (~11 GPU-h for 3 pairs) is moderate value now that onset has
nine pairs and two corpora. A judge-free head-to-head against the *metered* decoder remains
**impossible** and the reason is on record in `results/onset_prediction_verifiable.md`: TinyComma is
the only openly licensed anchor with the Llama-3 tokenizer and it scores `0.04` on GSM8K.

## What this session added to the record

Cautions stand at **twenty-seven**; three were added earlier in the session — **(y)** `**text**` is
markdown and renders as literal asterisks, **(z)** a `\label` in both body and appendix silently
redirects every `\ref`, **(aa)** `served_prompt()` reconstructs the judge's prompt and disagrees
across arms in 455 of 500 cases, with judge C disclosed as the opponent's own checkpoint.

feat-116 and feat-117 added no new caution, because every trap they could have hit was already
written down. What they added is the clearest worked example this project has of the rule the
cautions exist to serve: **a shape you can see in a table is not a finding until an interval says
so.** feat-116 read a curve off a table and drew a deployer-facing warning from it; feat-117
registered the test and the warning did not survive; the appendix says so rather than quietly
dropping it.

Three habits, all earned the hard way here:

- **Render the page and look at it.** A figure font change made to buy page space collided the axis
  labels and put the legend on the data; the build was clean and every automated check passed.
- **Run the *full* gate after adding any float.** A new appendix table produced a 52.8pt overfull
  box that the page-budget check does not see.
- **Measure a parameter count, never read it off a model's name.** `Qwen2.5-7B-Instruct` is
  `7.6156`B and pricing it at `7.0` understated a headline number by `8.8%` for two commits.
