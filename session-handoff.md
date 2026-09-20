# Session handoff --- 2026-09-20

## Current objective

**Revision of `iclr_2027.tex` against four referee reports** (one AC/PC LLM report, three full
reviews). The reports are consistent on one thing above all --- the paper's headline impossibility
claim outruns its own theorem --- and that is what most of this session did. The manuscript is
current: main text **exactly 9 of 9 pages** (Ethics Statement at the top of page 10 with no body
prose above it), **0 overfull**, **0 `??`**, 34 pages total, `pdffonts | grep -ci bold` = 3.

## Arms in flight on the DGX (2026-09-20, end of session)

One, and one closed. **`results/onset_prediction_lambada_headtohead.md` is SCORED and INVALID**
--- a second judge-free head-to-head on LAMBADA, the task the one meterable anchor can actually do
(it scores `0.04` on GSM8K and below chance on MMLU). Its H1a instrument gate caught the arm: the
UNCONSTRAINED risky model read `0.102 [0.076, 0.130]` against a registered floor of `0.40` on a
task whose published number is about `0.70`. The cause is the **echo** --- both models repeat the
prompt's tail before continuing, so the gold word is in the completion but is not its first word,
and `extract_lambada` takes the first word. **That is the same defect that killed both MMLU arms,
and the pre-registration claimed the parser had been validated against output shapes.** It had
been pinned to five shapes in `tests/test_lambada_extraction.py` and every one of them was
HYPOTHESISED; no generation had been read. Recorded as **caution (au)** in `AGENTS.md`, and
`tests/test_lambada_not_quoted.py` (4 tests) fails if the paper ever reports LAMBADA or stops
describing the judge-free head-to-head as a one-task result. Per the registered H4 there is no
second repair on this corpus.

**`results/onset_prediction_tqa_vacuity.md` is SCORED** --- `S(x)` for a TriviaQA answer under the
anchor, the quantity `sections/appendix_selection.tex` needed and never had. That paragraph called
`72` nats ``already vacuous'', called `12` and `24` ``the budgets whose certificate is not
vacuous'', and then closed by saying no vacuity number came from the arm --- three claims with no
measurement behind any of them, the first two contradicting the third inside one file
(caution (ao)). No generation: the anchor and the risky model are teacher-forced
on the gold aliases under the prompt `h1.py` actually served, read verbatim from
`data/bench/triviaqa_factual.jsonl`. All three instrument gates passed and
**two of four bands were REFUTED**: the anchor's median `S(x)` is `5.86` nats against a smallest
metered budget of `12`, so the metered certificate is vacuous on `89.6%` of the questions at
`k=0.5` and `100%` at `k=3` --- ``the budgets whose certificate is not vacuous'' names an empty set
and is withdrawn from the manuscript. Reported with it, post hoc and symmetric because omitting it
would be one-sided: selection's `log n` is vacuous on `35.4%` of the same questions at `n=64`.

**Every arm launched THIS SESSION has landed and is scored. One older arm is still running:
feat-134**, the Comma-7B `n=128` ceiling (`results/onset_prediction_comma7b_n128.md`, bands
committed at `3a9c8d5`, the arm that was explicitly asked for because it is over the 24-GPU-hour
escalation threshold). State as of 2026-09-20 19:55: **creative and factual are generated**
(`factual` has its `GEN_DONE`; `creative` finished `rc=0`, and card2b writes no sentinel for its own
class), **neutral is on attempt 5** --- the earlier four were killed by another project's jobs on
this shared box --- at `2200/25600` after 89 minutes, so about **15--16 hours remain**.

**Two coordination faults were found and one was repaired.** `run_comma7b128_card2b.sh` is waiting
on neutral's sentinel with a **12-hour** budget and was at `10500s`, so it aborts about seven hours
BEFORE neutral finishes. `run_comma7b128_merge.sh`, whose budget is **48 hours** and which exists
precisely for this, had been **dead for 31 hours** (log frozen at `13:06` the previous day). With
both gone nothing would have merged or scored the arm and the whole 30 GPU-hours would have
produced no result. The merge shell was relaunched on GPU 2 at `19:53` (PID reparented to init, log
`output/logs/comma7b128_merge.log` tracing every 60s); it allocates no GPU until the sentinel
appears, so it costs nothing while it waits. Card2b may still abort --- that is harmless now, and a
double merge would have been harmless anyway (same inputs, same command, same output paths).

**feat-135 is CLOSED: CLIMBS (MARGINAL), NOT PROMOTED.** Its stage-2 generation and scoring had
in fact completed at `11:25` on 2026-09-20 while its log still said "stage 2 has NOT been started",
so the arm was sitting finished and unread. All four gates pass (G1 vetting `0.0000`; G2 the full
seven-point grid on `500` prompts; G3 empty fraction **`0/500`**; G4 **`72.1`** words), and
`analysis/score_kl3m37b_breadth64.py` --- written and mutation-tested before the data --- reads
paired `g(64)-g(8) = +0.0580 [+0.0200, +0.0940]`, judge~B. The interval excludes zero, but at
**`1.57`** interval half-widths it is below the `2.0` the registration fixed in advance, so it is
MARGINAL and **the appendix keeps its two-anchor statement**. The second judge disagrees outright
(`+0.0300 [-0.0120, +0.0710]`, SATURATED), which strengthens that call.

**The seed replication the marginality rule demands already exists on host B and does not support
the climb**: `kl3m37bhb64` reads `+0.0200 [-0.0140, +0.0540]` and `kl3m37bhb64s62`
`+0.0130 [-0.0230, +0.0500]`, a clean within-host pair differing only in seed, agreeing to `0.007`,
both containing zero. It is cross-referenced in feat-135's log and **not pooled** with the local
reading (caution (ap)), and the local-vs-host-B gap is not evidence against the local number because
it changes silicon and seed together (caution (at)). Caution (ap)'s half-width rule predicted
DOES NOT REPLICATE at `1.57` and that prediction held out of sample.

**`results/onset_prediction_cotaeval_news.md` (feat-155) is IN FLIGHT on host B** --- the
Program Chairs' one remaining partially-answered point, that the method is benchmarked only on our
own CopyBench/BookMIA setups and not on the community-standard **CoTaEval**. It is a domain gap as
well as a framework one: every protected corpus in this paper is books, and this is news. Primary
axis is CoTaEval's own in-domain utility, `500` article+question items at Comma-7B, `n` to `64`,
scored by **SQuAD token-F1** --- graded, and chosen because six anchor generations were read before
any band was written and they show partial answers (`' On the morning of May 25'` against gold
`'May 25 , 1979'`) that exact match would score zero. Registering exact match would have repeated
the MMLU/LAMBADA defect for a third time. The `1000`-item infringement half runs too and is
explicitly a **negative control, not CoTaEval's memorization setting** --- neither model has seen
NewsQA. Bands, four instrument gates and a `2.0`-half-width marginality rule are committed.

**`results/onset_prediction_cotaeval_breadth.md` (feat-156) is IN FLIGHT on host B**, filling the
other seven H100s: the same CoTaEval news protocol at **seven anchors in three families**, plus a
disjoint-seed re-draw of the headline anchor. One setup is the weakness the paper's own breadth work
exists to fix, so the CoTaEval answer should not itself be a single setup. Gates are applied per
anchor and never pooled, and **G1 is expected to fire at the small anchors** --- that is the
capability floor the paper already concedes, not a failure of the mechanism. The replication's
prediction is registered **conditionally on feat-155's half-width ratio, before either number
exists**, which is what stops caution (ap)'s rule being fitted after the fact.

**`results/onset_prediction_cotaeval_scorer.md` (feat-157) is SCORED: SCORER-BOUND, and we
predicted SCORER-INDEPENDENT.** All four `14`B-scorer arms landed on host B and were scored with
`analysis/score_cotaeval.py --scorer-scale`. The turn-over **survives** at Comma-7B (2T) and at its
disjoint re-draw (`3.09` and `3.18` interval half-widths) and **disappears** at Comma-7B (1T) and
TinyComma-1.8B --- two of four, which the table fixed before the run calls SCORER-BOUND. The rescue
is narrow and the appendix says so: neither anchor that stopped losing began to climb, both read NO
EFFECT and both are MARGINAL, so a larger scorer buys the absence of harm and not a gain, and it
does not reach the anchor the headline uses. **The main text is therefore unchanged** --- ``loses F1
on CoTaEval'' is true of the headline anchor at both scorer sizes --- and the rescope is one
appendix paragraph stating the `7`B requirement explicitly. G0, the instrument gate this
registration added, passed at all four with the `n=1` F1 *identical to four decimals* to its
counterpart's, the tolerance derived by the scorer from each counterpart's own CSV rather than
typed in, and mutation-tested five ways **before** it was run. `tests/test_cotaeval.py` gains five
guards, mutation-tested nine ways; two patterns did not land first time (caution (ar)) and one
mutation found a real gap --- a sign-flipped gain fired nothing until verdict-and-sign agreement was
asserted.

**The AC/PC report is fully audited and three remaining items are closed**: the `s_r` column is
defined in its caption, the Appendix F window factor has its formula (from `analysis/order_law.py`,
not described), and the Conclusion's hardest sentence is split length-neutrally with every guarded
literal kept. One item is deliberately open --- the evaluation-parity paragraph (W4's first half);
its substance is answered in four places but not framed as one paragraph, and seating it at a
zero-slack 9-page body would cost a concession (caution (ag)).

**Next session: check `output/logs/comma7b128_merge.log` first.** If neutral's `GEN_DONE` never
appears, the supervisor (`run_comma7b128_supervise.sh`, PID reparented, 27 h old) is still
restarting it against the other project's memory pressure; the card must have `34000 MiB` free.

The last to close was **`results/onset_prediction_second_opponent.md`**, and it is the one
robustness check in this paper that **did not survive**. `Qwen2.5-14B-Instruct` replaced the fixed
opponent, judge~B re-scored the same four committed arms, nothing else changed. Both mechanisms
still beat their own controls (`+0.0540` and `+0.0605`, both clear of zero); the difference between
them is `-0.0065 [-0.0385, +0.0255]`, `0.20` interval half-widths from zero, against
`+0.0645 [+0.0300, +0.0995]` on the registered pass. The registered taxonomy keys on the interval,
so the reading is **UNRESOLVED**, and its pre-fixed consequence was applied verbatim: the abstract
now says ``four of five judges **against one fixed opponent**'' and Section 6's heading reads
``It replicates across judges, not across opponents''. The certificate and the judge-free axis are
untouched --- neither has an opponent --- which is why the paper rests on them. We predicted
REVERSAL HOLDS and were wrong; that is recorded rather than reframed.

Closed today:

| log | reading |
|---|---|
| `onset_prediction_frontier_judge.md` | judges D, E split; the family attribution in its own H3 is **falsified** |
| `onset_prediction_judge_panel.md` | **PANEL CONFIRMS, 4 of 5**; family, size and consistency all fail to order it |
| `onset_prediction_memfree_headtohead.md` | H2 PARTIAL (by `0.000059`), H3 **BLOCKLIST HOLDS** (our prediction refuted), H1 INCUMBENT WINS **passed by a no-op** |
| `onset_prediction_cpfuse_headtohead.md` | H1 REPRODUCES; H2 **withdrawn, not scored**, with the reason recorded |
| `onset_prediction_mmlu_headtohead.md` | **INVALID** --- two defects of ours (parser, undirected gate) |
| `onset_prediction_mmlu_rescore.md` | **BELOW CHANCE**; no number enters the paper, enforced by a test |
| `onset_prediction_mmlu_comma7b.md` | H1 pass, H2 **SELECTION LIFTS**, H3 REWARD TRACKS --- the third judge-free task |
| `onset_prediction_multilingual.md` | **SAME PICTURE** --- admissible adversary (`0.5455`, stronger than English), selection `0.0000` at every `n`, uncontaminated anchor |

## Recommended next step

Nothing is blocked and nothing is in flight. The paper answers every empirical ask in the four
reports. Three things a next session could take up, in order of value:

1. **A fourth judge-free task with a meter in it.** The judge-free head-to-head is still ONE task
   (TriviaQA), because TinyComma is the only anchor a metered decoder shares a vocabulary with and
   it cannot do GSM8K or MMLU. Finding a task it *can* do is the single largest remaining gap, and
   Limitations says so plainly rather than implying a plural axis.
2. **Judged utility in French or German.** `onset_prediction_multilingual.md` establishes
   extraction outside English and explicitly does **not** claim the anchor serves good French;
   that is the open question it leaves behind.
3. **A second risky family.** Every judged arm uses `Llama-3.1-8B-Instruct` as the fixed opponent.
   `Qwen2.5-14B-Instruct` is cached and could be generated against.

Do **not** re-run the MMLU head-to-head at the audited anchor: its registered band says there is no
third attempt, and `tests/test_mmlu_not_quoted.py` enforces that no number from it reaches the
paper.

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
