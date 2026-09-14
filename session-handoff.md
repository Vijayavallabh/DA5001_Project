# Session handoff — 2026-09-15 (early morning, after feat-116)

## Current objective

**In flight: feat-117**, `results/onset_prediction_scorer_scale.md`, GPU 0, launched 2026-09-15
03:5x, roughly two GPU-hours. It asks where between `0.5`B and `7.6`B best-of-n stops turning over,
by re-scoring the same cached candidates with `Qwen2.5-1.5B-Instruct` and `Qwen2.5-3B-Instruct` and
judging all four scorers in one pass. Bands G0-G4 committed at `cb660e8` before the run; G0 is a
replication gate against feat-116 and if it fails nothing else may be quoted. If this session ends
before it lands, the log is `output/logs/scorer_scale.log` and the outputs are
`results/scorer_scale{,_bands,_per_prompt}.csv`.

All forty-four other pre-registrations in `results/` are scored and the manuscript compiles clean
from a deleted PDF.

This session did four things: **reframed the paper to lead with its contribution (v8)**, **answered
a referee report by measurement (v9/v10)**, **read the rendered PDF end to end**, and **ran the
paper's own escape hatch from the compute concession and reported that it failed (feat-116)**.

### feat-116 — the 0.5B scorer does not carry the gain

`serving_cost.csv` carried a row marked `0.5B, NOT RUN`: the arithmetic saying that because the 7B
reward model dominates the price, a small scorer would cut `57.5x` to `15x`. An unmeasured
counterfactual is the thing this paper exists to object to, so it was pre-registered
(`results/onset_prediction_compute_matched.md`, five bands, committed to git at `1cc982d` **before**
the run) and run: `Qwen2.5-0.5B-Instruct` (`0.494`B measured), identical template and Yes/No reward,
re-scoring the same 32,000 cached candidates, then both scorers judged over the whole nested grid
under feat-113's corrected protocol — 8,260 calls, both orders, one true prompt, one fixed opponent,
`0.23` GPU-h.

**F5 replicates first** (`sel7b_n64` `+0.1075` against feat-113's `+0.1045`; `metered_k10` `+0.0400`
exactly), which is the gate that licenses quoting anything else. Then: **F1 FAILS**
(`+0.0220 [+0.0000,+0.0440]` at `n=64`, against a committed `WORKS` in `[0.03,0.09]`), **F2 COSTLY**
(`-0.0855`, worse than the band I wrote), **F4 MATCHED-COMPUTE LOSS** (`-0.0395 [-0.0720,-0.0065]`
at `0.94x`, against a committed `PARITY`), **F3 NO CROSSING**. Wrong on three of four.

F3 needs its margin, not its label: at `n=16` and `3.75x` the small scorer reaches `+0.0395` against
the meter's `+0.0400` — **five ten-thousandths** short, intervals almost entirely overlapping. The
honest reading is *indistinguishable from* the meter; the registered rule asked for above.

**The unregistered finding is the one worth carrying forward.** The 7B scorer is monotone in `n`
across the grid; the 0.5B scorer **peaks at `n=16` and falls**, ending below where it stood with a
quarter of the draws. Taking the argmax of a weak score over a larger pool selects increasingly on
its noise, so `log n` is not a free knob — the certificate keeps improving in `n` while the utility
bought with it turns over. `compute_matched_scorer_agreement.csv` (post hoc, no band) says why
without a judge: Spearman `0.1333` between the two rankings within prompt, same served draw on
`0.052` of prompts at `n=64` against `0.016` by chance.

Committed consequence applied: **`57.5x` stands exactly as written** in the introduction, Section 2
and Limitations, and Limitations carries the committed sentence — the gain is the *scorer's*
capability, not the mechanism's, and the cost is intrinsic at the scales tested. One declared
departure from the letter, reasoned in the scoring log: the small-scorer rows stay in
`serving_cost.csv` rather than being deleted, because they are now a measured negative and deleting
one would hide it. What is gone is the framing they were written to support.

The arm also forced two corrections it was not built to find: an **estimand mix in Limitations**
(`+0.054` single-order quoted against `+0.040` order-averaged — the read-through's Table 1 defect,
surviving in the Limitations paragraph, pinned by no test; the true figure is `+0.0415`, so at
`7.2x` selection *matches* the meter rather than beating it), and the first **measured** crossing
for the 7B scorer, `n=8` at `7.18x`.

### v8 — the paper now leads with the result

v7 argued the dichotomy and reached selection anchoring on page 5, so a reader met four pages of a
mechanism failing before meeting one that works. The two roles are swapped. Section order:

1. Introduction · 2. **Selection anchoring** (was §5) · 3. **Does it work?** (was §6) ·
4. **Why a per-token budget is vacuous or trivial** (was §2) · 5. **The deployed instance, and the
three repairs it has left** (was §3+§4; `onset.tex` is now its opening paragraph, `\input` from
`orders.tex`) · 6. Related work · 7. Limitations and conclusion

`prop:threshold` moved from `frontier.tex` into `selection.tex`, because what makes `log n` a
*certificate* is where it sits against `S(x)`. Propositions now number 1 selection, 2 threshold,
3 sparse, 4 imitation, 5 outrun (appendix), Theorem 1 nfl. Title is *Two Nats, Not Two Thousand:
Spending the Copyright Budget on the Draw Instead of the Token*. Every v7 section is kept beside its
replacement as `*_v7_2026-09-14.tex`. **No measured number changed in v8.**

### v9/v10 — four referee objections, fixed in substance

The review scored 5/10 reject. Four points were correct:

1. **The composition silently swapped Rényi order.** The certificate is advertised at
   `D_inf = log n` and the odometer count was priced at the KL bound. Both orders are named now and
   the conservative one leads: **192** queries pathwise (max-divergence composes additively, so it
   needs no advanced composition theorem — the direct answer to Cohen 2025), **332** at KL, against
   **2** for the meter at realised spend and **0** at its own published certificate, since one
   200-token response is certified at 600 nats against a 400-nat odometer.
2. **feat-113, the order-averaged head-to-head.** The headline compared two gains neither of which
   was order-averaged, though our own instrument check calls the judge UNUSABLE. 4,000 judged pairs,
   four arms, both orders, paired over 500 prompts: **REVERSAL CONFIRMED**, `+0.1045
   [+0.082,+0.128]` against `+0.0400 [+0.014,+0.0655]`, difference `+0.0645 [+0.030,+0.0995]`.
   **My pre-registered prediction (UNRESOLVED) was wrong** — order-averaging moves selection by
   `0.95` and the meter by `0.41`, so the difference *grows* five-fold. Position bias was flattering
   the meter, plausibly because at `k=10` the meter nearly *is* the risky model it is judged against.
3. **feat-114, serving cost.** `results/serving_cost.csv`: selection is `7.2x` the metered decoder's
   forward-pass cost at `n=8` and `57.5x` at `n=64`. The reviewer's 30–60x estimate was right and
   the nats axis hid it. In Section 2 and in Limitations.
4. **feat-115, anchor vetting.** `Pr_{p_s}[E]` is the one quantity a deployer can measure outright.
   Over eighteen models it separates completely: five openly licensed anchors read `0.000` on all
   100 passages, twelve contaminated read `0.570`–`0.980`, and `Llama-3.1-70B`, which memorised in
   pre-training, reads `0.500` with one passage reproduced in full. **Post hoc** — no band was
   committed, and the file is deliberately *not* named `onset_prediction_*` so it cannot inflate the
   pre-registration count. Answers Q8 and W2; the Ethics Statement now says *how* to vet an anchor.

Also: Section 4 retitled off its overclaim (Proposition 3 constrains *shape*, not achievable
utility); the CP-Δ/CP-k scope limit stated; judges named (A Qwen2.5-7B, B Phi-3.5-mini, C
Llama-3.1-8B) with **judge C disclosed as the opponent's own checkpoint**; the tie structure
explained; the anchor-scale ceiling attributed to the licensing frontier; onset softened to what
nine pairs can carry; Kalai et al. and Chen et al. positioned as prior art rather than scenery.

**Deliberately not addressed:** the "theory is elementary" criticism. The paper already concedes the
inequality is elementary and cites Beirami for the KL form; narrowing further would understate what
is new, which is the *reading* of that inequality as a copyright certificate comparable to the
deployed meters on one divergence axis.

### The read-through — two estimand defects the fixes themselves introduced

Read all nine body pages and the new appendix material as rendered rather than as source:

- **Table 1 mixed estimands.** Its four selection rows were single-order while the metered row had
  been switched to the *order-averaged* `+0.040` in v9. Metered row restored to its single-order
  `+0.072`; the caption now states the table is single-order throughout and that the order-averaged
  head-to-head in the text is the comparison the paper stands behind. Appendix I reconciled with it.
- **A stale claim in Section 2** — "the two mechanisms reach the same judged utility on this
  workload", true when both read `u ~ 0.52` and false since feat-113. Cut.
- Five smaller: the contributions bullet still promised what Section 4 was retitled away from; the
  abstract and conclusion had been trimmed to a bare `8` with "amplification" cut away; two
  compression artefacts in the introduction; a caption describing a dash that no longer appeared.

## State

| | |
|---|---|
| manuscript | `~/sub/satml/iclr_2027.tex`, **9 of 9 body pages**, 53 total |
| build | exit 0, **0** overfull, **0** unresolved, **0** literal `**`, page 10 body-free |
| tests | **440 passed**, `./init.sh` exit 0 |
| pre-registrations | **44**, all scored (`onset_prediction_compute_matched.md`, `onset_prediction_order_averaged_h2h.md`) |
| numeric audit | 2,762 literals, 1 expected miss (`64256`, the Comma-7B padded embedding count) |
| compute | 221.4 measured over 232 jobs, "approximately 221" disclosed |
| artifact | 813 files, `MANIFEST.sha256` verified |
| anonymity | 0 "our earlier audit", 0 affiliation; name appears only as `(Vijayavallabh, 2026)` |

## Files changed this session

Manuscript: every body section rewritten or reordered, plus `appendix_selection.tex` (three new
paragraphs — the order-averaged head-to-head, the composition table at both orders, anchor vetting),
`appendix_proofs.tex` (duplicate label removed, threshold proof generalised to an arbitrary law),
`appendix_related.tex` (Kalai/Chen), `references.bib` (+`panickssery2024llm`).

Repo: new `analysis/order_averaged_h2h.py`, `analysis/serving_cost.py`, `analysis/anchor_vetting.py`;
new `results/order_averaged_h2h{,_per_prompt}.csv`, `serving_cost.csv`, `anchor_vetting.csv`,
`onset_prediction_order_averaged_h2h.md`; `scripts/snapshot_manuscript.sh` now follows nested
`\input`; `tests/test_selection_claims.py` and `tests/test_imitation_cost.py` updated.

## Recommended next step

**Where does monotonicity come back?** feat-116 leaves one sharp, cheap question open. A 7B scorer
is monotone in `n` and a 0.5B scorer turns over at `n=16`; nobody knows where between them the
turnover appears, and that boundary is the deployer-facing number this paper would be the first to
give. Two more scorers on the same cached candidates — `Qwen2.5-1.5B-Instruct` and
`Qwen2.5-3B-Instruct`, the same family so scale is again the only variable — would place it.
Re-scoring is `0.2` GPU-h and the judging is the same 8,260-call pattern, so roughly `0.5`–`1`
GPU-h total, no generation. It needs its own pre-registration with a band on where the turnover
falls, and it should commit in advance to reporting a monotone 1.5B arm as "no turnover found
between 1.5B and 7B" rather than searching for a scale that turns over.

This matters beyond the frontier: the paper's claim is that the certificate is the mechanism's and
the utility is the scorer's, and a measured scorer-scale boundary is the strongest form of that
claim. It also gives Limitations a number where it currently has a direction.

Second, unchanged: **BookMIA-50 onset** (~11 GPU-h for 3 pairs) is moderate value now that onset has
nine pairs and two corpora. A judge-free head-to-head against the *metered* decoder remains
**impossible** and the reason is on record in `results/onset_prediction_verifiable.md`: TinyComma is
the only openly licensed anchor with the Llama-3 tokenizer and it scores `0.04` on GSM8K.

## Traps this session added to AGENTS.md

Cautions are now **twenty-seven**. This session added three:

- **(y)** `**text**` is markdown, not LaTeX, and renders as literal asterisks. Five reached the
  compiled PDF; no compile-time check sees it.
- **(z)** a `\label` set in both the body and an appendix restatement silently redirects every
  `\ref` to the appendix. `prop:sparse` did this from v5 to v7, so the introduction's reference to
  the paper's own dichotomy pointed at an appendix restatement numbered 6 while the body's statement
  was 3 and nothing referred to it. Exit 0, 0 overfull, 0 `??` — the number that renders is a real
  proposition number, just the wrong one.
- **(aa)** `served_prompt()` reconstructs the judge's prompt as *served text minus generation*, so
  it disagrees across arms for the same `prompt_id` in **455 of 500** cases. Under one true prompt
  the single-order head-to-head difference is `+0.013`, not the `+0.070` the published pair implied.
  Any new judged arm must take the prompt from `dap.shared.load_prompt_corpus`. The same entry
  records that **judge C is the risky model's own checkpoint**.

feat-116 added no new caution: every trap it could have hit was already written down, and the two
habits below are what caught the page-budget work. What it did add is a *finding* worth a caution's
weight if it reproduces — `log n` is not a free knob, because a weak scorer's argmax over a larger
pool selects on its noise, so the certificate improves in `n` while the utility turns over. It is
one scorer at one scale and is recorded in the appendix as an observation, not a law.

And two habits, both earned the hard way here:

- **Render the page and look at it.** A figure font change made to buy page space collided the axis
  labels and put the legend on the data; the build was clean and every automated check passed.
- **Run the *full* gate after adding any float.** A new appendix table produced a 52.8pt overfull box
  that the page-budget check does not see. Three consecutive passes each introduced a defect while
  fixing another — the label collision, the estimand mix, the overfull table. Tests caught the
  first; only reading the rendered page caught the other two.
