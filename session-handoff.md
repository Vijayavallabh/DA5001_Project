# Session handoff --- 2026-09-20

## Current objective

**Revision of `iclr_2027.tex` against four referee reports** (one AC/PC LLM report, three full
reviews). The reports are consistent on one thing above all --- the paper's headline impossibility
claim outruns its own theorem --- and that is what most of this session did. The manuscript is
current: main text **exactly 9 of 9 pages** (Ethics Statement at the top of page 10 with no body
prose above it), **0 overfull**, **0 `??`**, 34 pages total, `pdffonts | grep -ci bold` = 3.

## Arms in flight on the DGX (2026-09-20)

Three pre-registrations are committed and **not yet fully scored**. Each has its bands fixed before
its run; none may be read until its arm lands.

| pre-registration | arm | state |
|---|---|---|
| `results/onset_prediction_mmlu_comma7b.md` | MMLU judge-free at Comma-7B, GPU 3 | generating |
| `results/onset_prediction_memfree_headtohead.md` | MemFree decode-time blocklist | **H2/H3 scored**; H1 (judged utility, ordinary workload) still generating on GPU 4 |
| `results/onset_prediction_cpfuse_headtohead.md` | CP-Fuse vs selection, GPUs 6--7 | shards trained and admissible (sampled nv-recall `0.905`/`0.833`); audit running |

Scored this session and closed: `onset_prediction_frontier_judge.md` (judge panel, first three),
`onset_prediction_judge_panel.md` (**PANEL CONFIRMS, 4 of 5**), `onset_prediction_mmlu_headtohead.md`
(**INVALID**, two defects of ours) and `onset_prediction_mmlu_rescore.md` (**BELOW CHANCE** at its
repaired gate; no number from it enters the paper, enforced by `tests/test_mmlu_not_quoted.py`).

## What was done

### 1. The impossibility claim was rescoped, everywhere it appears

All four reports make this their first or second soundness point. Proposition~3 bounds
`E[N_eps] <= K/eps` for any law with sequence KL `<= K`. It therefore says a *bounded* budget must
be **sparse**; it does **not** say a sparse policy is useless, and it does not exclude a causal
policy that concentrates. The universal the paper may claim is about a **rate**, which issues an
allowance at every step and so has `K = Omega(T)`.

* Abstract, introduction, `frontier.tex`, `orders.tex`, `iclr_closing.tex`: "no per-token
  **budget**" -> "no per-token **rate**"; "Vacuous, or trivial" -> "**Vacuous, or sparse** --- and
  sparse is a shape, not a verdict."
* The reason it is a shape and not a verdict is **our own mechanism**: selection's served law
  factorises like any other, so Proposition~3 applies to *it* at its own `3.175` nats, and it gains
  `+0.1045` out of that budget. That is now stated in both the introduction and Section~3.2.
* `sec:orders` now says the sparse causal construction's `+0.0545` is "an empirical shortfall for
  the best construction we could build, not a proof that none matches", and the Conclusion keeps
  the class-level question open in those words.

### 2. Four mathematical defects the AC found, all real, all fixed

* **Proposition~1's tie hypothesis was superfluous** and contradicted its own proof. It now holds
  for *any rule that serves one of the draws* --- which is strictly stronger, covers majority vote
  with data-dependent ties rigorously (Review 3's request), and leaves the sharper KL form scoped
  to the tie-free argmax, where it belongs.
* **Proposition~4's statement bounded `Z_T` over ALL steps while its proof restricts to the slack
  set `S`.** The statement now matches the proof and carries the missing "slack steps carry
  divergence bounded away from 0".
* **The `Omega(T)` overhead silently assumed `Lambda*_s(u_max)` is constant in `T`**, which fails
  for a utility only an exponentially rare sequence attains. The proviso is now in the proposition,
  and the appendix says what happens without it (`Theta(T)/Theta(T) = Theta(1)`). The utility this
  paper measures satisfies it and is measured: `1.30` nats, unchanged by length.
* **Proposition~5's equality condition was backwards.** With `delta > 0` a constant surprisal rate
  makes the running ratio `s(x) + delta/N_t` strictly *decreasing*, so the maximum is at `t=1` and
  the inequality is strict. Corrected, with the `delta = 0` case stated separately.
* Also: the **duplicated `\subsection{Proposition 5}` header** is gone (proof merged into the
  statement's subsection), and a **truncated paragraph beginning "margin: at k = 20 ..."** --- a
  page-trim artefact that reached the compiled PDF --- was rewritten.

### 3. "Selection escapes the chain rule" was WITHDRAWN as an error of ours

Appendix~A argued selection "is not a causal policy at all" and that `N_eps = 0` identically. That
is a statement about the *sampler*; the served law `q` factorises like any other and its
conditionals do differ from `p_s`. The withdrawal is recorded in place (so it can be audited) and
replaced with the operative difference: the conditionals are not **causally computable**, because
`q_t(. | y_<t)` depends on the scores of *complete* candidates.

### 4. The incumbent baseline four reports asked for

`results/blocklist_{data,bookmia100,bookmia_all}.csv` already held a measured MemFree n-gram rule
and the paper never used it. Main-text Related Work now carries a **Scope** statement (certified
divergence budgets vs. blocklists / datastores / filtering, with CoTaEval named), and
`appendix_related.tex` has the measurement --- including that it **refuted the argument it was
built to support**: collateral on ordinary text goes `0.000% -> 0.140% -> 0.140%` while the index
doubles to `2.24M` n-grams. A blocklist's collateral is *bounded*; that claim is withdrawn.

### 5. The capability gap, promoted out of a clause in the Conclusion

`results/verifiable_metered_tqa.csv` is a pre-registered, judge-free, matched head-to-head at the
one anchor a meter can be run at, and the paper gave it one clause. It is now **panel (c) of
Figure~3**: the metered decoder is the anchor's own `0.112` at `k=0.5` and `k=1` and reaches
`0.618` only at `k=20`, where it *is* the risky model, certified at `480` nats for a `24`-token
answer. The arm we lose is now the clearest picture of the dichotomy in the paper.

### 6. Smaller things

Relative-not-absolute caveat on "reproduces no protected passage" in the abstract and the
contributions; prompt-set provenance (`200` general-knowledge, `150` FactScore biography, `150`
creative-writing --- **verified against the run's own records**, an earlier draft of this sentence
invented it); the scorer model named where the scorer is introduced; Theorem~1 restated for *any*
`K`-bounded law (Review 4 Q8), which let Eq.~(1)'s closed form move to the appendix; the two
appendix tables a reader is sent to are now numbered and captioned (`tab:imitation`,
`tab:imitation-pairs`) and Appendix~A's empirical half has its own subsection; the extraction
table says its floats are means and that a dash is *not measured*; `arXiv preprint arXiv:NNNN` ->
`arXiv:NNNN` across 31 bib entries.

### 7. Page budget

Every addition was paid for. `safety_envelope` moved to `appendix_selection.tex` (one float is
worth ~19 body lines, measured by deleting it and rebuilding) with **all of its numbers carried
into Section~4.3 prose**; Eq.~(1) moved to Appendix~A; the two body figures were narrowed to
`0.84` and `0.93\textwidth` (shrinks `0.780` and `0.881`, both well above the `0.70` floor of
caution (af)); captions and duplication trimmed throughout.

## Guards

`tests/test_review_revisions.py` is new: 13 guards over the rescoping, Proposition~1's
strengthening, Proposition~4's two conditions, Proposition~5's equality condition, the new figure
panel, the blocklist claim *shape* (it must flatten, not merely have those cells) and the
relative-guarantee caveat. **All seven of the load-bearing ones were mutation-tested** --- break
the claim, confirm a named test fails, restore byte-identical.

**Fifteen existing guards fired during the trim and every one of them was right.** Each was a
committed claim or concession my compression had removed (the odometer's `1.204`/`332`, "one
fifty-fourth", "at all six anchors", `rho = +0.543` "on the two-judge mean", "inseparable from a
no-effect null", "$-0.019$ for $60.6$", the imitation rates `0.617/0.303/0.908`, "where that bar
sits is open", the `3.4x` GSM8K ratio *with its n*, "four draws ... at this anchor, eight at its 1T
sibling", `prop:sparse` in Limitations). All restored. Caution (ag) held exactly as written: a
length edit deletes concessions first.

One guard was **repaired rather than satisfied**: `test_collapse_robustness_prose.py`'s
no-separation check matched two exact phrasings and the rescoping broke both with the substance
untouched (caution (aj)). It now finds every budget the abstract names inside a no-separation claim
and checks each against `judge_separation_v6*.csv`. Mutation-tested three ways.

## Recommended next step

1. Read the compiled PDF end to end. The theory sections were rewritten, not patched, and this
   session did not do a full read-through of the rendered pages beyond 2, 9, 10 and 15.
2. Six `center+tabular` displays in `appendix_robustness` and `appendix_selection` are still
   unnumbered. This was a **deliberate** decision, not an omission: each completes the sentence
   above it (they follow a colon), nothing `\ref`s them, and floating them would detach them from
   the prose they belong to. If a referee presses, the fix is to float them with captions.
3. The reports ask for things this session did not attempt and could not: a CoTaEval run, a
   CP-Fuse head-to-head, human labels, a frontier-class judge, and a realised (rather than
   closed-form) selection divergence. The last of these is the cheapest and is partly answered
   already --- the contaminated-anchor arm *is* the measurement of realised amplification against
   the `log n` bound, and Section~3.3 now says so.

## In flight, untouched by this session

This session edited only the manuscript, the figure script and `tests/`. It started no GPU work and
stopped none. The eight pre-registrations below are committed with bands and **not yet scored**;
they are listed here because an unscored registration is an arm that was committed and never
reported, which is fine only while it is running (`tests/test_preregistration_count.py`).

What was **verified** at the time of writing, by process list and `nvidia-smi`, is only this: two
`h1.py` generations against `common-pile/comma-v0.1-2t` are alive, at **7 h 20 m** and **6 h 40 m**
elapsed, and GPUs 0, 1 and 4 are busy while 2 is free. Their mapping onto the registrations below
was **not** re-derived, and the next session should read the run directories rather than trust this
paragraph.

| registration | what it asks |
|---|---|
| `onset_prediction_comma7b_n128.md` | where Comma-7B's ceiling is (feat-134) |
| `onset_prediction_kl3m37b_breadth64.md` | does the climb to `n=64` survive outside the Comma family (feat-135) |
| `onset_prediction_breadth_ladders.md` | two within-family capability ladders, second host (feat-136) |
| `onset_prediction_breadth_seed_replication.md` | do the ladder's verdicts survive a fresh draw |
| `onset_prediction_within_host_spread.md` | is a host change larger than a within-host re-draw (feat-141) |
| `onset_prediction_single_host_ladder.md` | the complete non-Comma ladder on one host (feat-140) |
| `onset_prediction_verifiable_comma1t.md` | does the judge-free climb survive a change of anchor |
| `onset_prediction_judgefree_offcomma.md` | is the judge-free climb Comma-specific |

Note that `feat-141` has already **withdrawn** `feat-140`'s central conclusion by the rule
registered before its data existed (commit `7c79a8d`), so the ladder registrations should be read
together and not separately.
