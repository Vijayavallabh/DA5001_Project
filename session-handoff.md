# Session handoff — 2026-09-15 (early morning)

## Current objective

Nothing is running. Tree is clean at `f619d6a` on `iclr-2027`, all forty-three pre-registrations in
`results/` are scored, and the manuscript compiles clean from a deleted PDF.

This session did three things: **reframed the paper to lead with its contribution (v8)**, **answered
a referee report by measurement (v9/v10)**, and **read the rendered PDF end to end**.

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
| manuscript | `~/sub/satml/iclr_2027.tex`, **9 of 9 body pages**, 52 total |
| build | exit 0, **0** overfull, **0** unresolved, **0** literal `**`, page 10 body-free |
| tests | **429 passed**, `./init.sh` exit 0 |
| pre-registrations | **43**, all scored |
| numeric audit | 2,762 literals, 1 expected miss (`64256`, the Comma-7B padded embedding count) |
| compute | 221.2 measured, "approximately 220" disclosed |
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

**A compute-matched arm.** The paper now concedes selection costs `57.5x` the meter at `n=64`, and
that concession is the weakest point a reviewer can still push on. `results/serving_cost.csv` has
the arithmetic that would fix it: the 7B reward model dominates the cost, and a **0.5B scorer**
would bring `n=8` to `1.9x` and `n=64` to `15x`. If a small scorer holds most of the gain, the paper
claims parity-or-better at comparable compute instead of conceding the axis. Needs its own
pre-registration with a band on how much gain may be lost, and roughly 2–3 GPU-hours to re-score the
cached `sel_anchor64` candidates — no generation, the candidates are on disk.

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

And two habits, both earned the hard way here:

- **Render the page and look at it.** A figure font change made to buy page space collided the axis
  labels and put the legend on the data; the build was clean and every automated check passed.
- **Run the *full* gate after adding any float.** A new appendix table produced a 52.8pt overfull box
  that the page-budget check does not see. Three consecutive passes each introduced a defect while
  fixing another — the label collision, the estimand mix, the overfull table. Tests caught the
  first; only reading the rendered page caught the other two.
