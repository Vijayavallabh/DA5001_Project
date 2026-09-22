# Session handoff --- 2026-09-22

## Current objective

**Revision of `iclr_2027.tex` against four referee reports** (one AC/PC LLM report, three full
reviews). The reports are consistent on one thing above all --- the paper's headline impossibility
claim outruns its own theorem --- and that is what most of this session did. The manuscript is
current: main text **exactly 9 of 9 pages** (Ethics Statement at the top of page 10 with no body
prose above it), **0 overfull**, **0 `??`**, tectonic exit `0`, `pdffonts | grep -ci bold` = 3.
**43 pages total as of 2026-09-22**, up from the 32 the 2026-09-19 reduction reached: the appendix
has taken this week's arms. That is content, not layout --- caution (ar)'s rule is that the
appendix's floor is set by the tables whose cells are guarded --- but if it has to come down again,
the lever is the newest additions and never a concession.

## Arms in flight (2026-09-22 16:25) --- ALL EIGHT HOST B CARDS BUSY

| card | arm | what |
|---|---|---|
| 0 | feat-172 Arm A | off-support ladder, AlpacaEval `n=256`; `84,525`/`206,080`, ~7 h left |
| 1 | feat-175 | opponent `Qwen2.5-0.5B-Instruct`, generating `700`/`850` |
| 2 | feat-174 | CoTaEval-QA draws |
| 3 | feat-172 Arm B | on-support ladder, `small` shard |
| 4 | feat-175 | opponent `Qwen2.5-1.5B-Instruct`, generating |
| 5 | feat-175 | opponent `Qwen2.5-3B-Instruct`, generating |
| 6 | feat-172 Arm B | on-support ladder, `factual` shard |
| 7 | feat-176 | Gutenberg completion workload, draws `500`/`32,000` |

**`results/onset_prediction_opponent_ladder.md` (feat-175) is REGISTERED and generating.** The
appendix concedes that the judged head-to-head *"does not survive"* a second opponent, on two
points, and the swap that produced it moved **family and size together** so nothing is identified.
Measured before the arm was designed, from per-prompt files already on disk: the committed opponent
beats the anchor control on `0.555` of prompts and `Qwen2.5-14B-Instruct` on `0.850`. **The
"second opponent" was not a perturbation** --- a `0.295` move on a bounded scale --- and
compression is a live, untested and far less damaging explanation for the whole result. Three more
opponents, **one family so size is the only variable**. H1 is on new data only: every opponent
measuring weaker than `0.555` must read REVERSAL CONFIRMED. H2 registers the ceiling at the other
end so it cannot be discovered. Eleven band and gate mutations pass, all written before any of the
three existed.

**`results/onset_prediction_fifth_workload.md` (feat-176) is REGISTERED and generating.** The last
candidate mechanism for the workload split. Anchor competence is refuted (feat-173 B4), the support
ceiling is retracted, and the prompt template is now **excluded** --- so the task TYPE is what is
left, and no completion workload we did not choose has ever been measured. `500` Gutenberg
excerpts through the factual slot, feat-170's protocol at `n=64`. **The training-data confound is
written down before the run**, with the rule that a win may not be used to revive the support story
it would appear to support: a loss here is decisive, a win is confounded and will be reported so.

**`results/onset_prediction_offsupport_ladder.md` (feat-172).** Arm A generating on card 0. Arm B
was **INVALID as first launched** --- its gate requires bit-identity against a reward cache built
on host B, and it was launched locally, where feat-136 measured only `17.6%` agreement. A
bit-identity gate is a constraint on the **silicon**, not only on the flags. Re-running on host B.

**`results/onset_prediction_fourth_workload.md` (feat-174) is REGISTERED and generating** ---
CoTaEval-QA, `500` prompts, the first workload that is neither instruction-following nor ours. Its
`G-cal` gate was amended before any band was read.

## Arms in flight (superseded, 2026-09-22 02:10)

**`results/onset_prediction_mixtral_power_k.md` (feat-168) is REGISTERED, calibration running** on
host B cards 0--4. feat-166 failed G0 because at `k=10` on AlpacaEval the budget is active on
`0.016%` of steps against `8.376%` on our own corpus and `794`/`805` served completions are the
unconstrained opponent byte for byte --- the arm labelled "metered decoder" was the risky model, so
feat-166 is INVALID per caution (w) and Mixtral's question is still open. feat-168 rebuilds the
instrument: a five-point `k` grid on `200` prompts, and the budget is chosen by `argmin` of
`|activity(k) - 0.08376|`, a rule fixed in the registration **before the sweep ran** so it cannot
be tuned to an answer. Bands carry over from feat-166 unchanged. The anchor draws, the opponent and
the `51,520` reward scores are NOT re-run --- they do not depend on `k`.

**`results/onset_prediction_third_workload.md` (feat-173) is REGISTERED and generating** ---
MT-Bench as a third workload, because two points cannot distinguish "instruction-following breaks
the reversal" from "AlpacaEval breaks it", and with two points every candidate mechanism fits. Full
feat-170 protocol at `n=256`, both the paper's `k=10` and a rate-matched binding budget. It also
registers, as explicitly **exploratory**, a predictor --- the anchor's own `n=1` win rate against
the opponent, measurable before any comparison --- and the per-class decomposition of our own
corpus, which gives three more workloads inside one pass at one protocol for no compute. If the
three classes span enough anchor competence to flip the sign within a single pass, that is stronger
evidence for the support-ceiling mechanism than any number of external benchmarks.

**`results/onset_prediction_mixtral_armc.md` (feat-171) is REGISTERED and judging** --- Mixtral,
third attempt, and the first on a workload where the comparison exists. feat-166 and feat-168 both
failed for the same reason we only understood this morning: both ran on AlpacaEval, where the
reversal does not hold, so judge~B read the difference negative and their gates correctly refused
to let Mixtral be read. feat-170 Arm C is `850` prompts at a binding budget with judge~B reading
`+0.0965`, and its generations and reward cache are on disk, so **only the judge changes**. Judge~C
runs beside it.

**`results/onset_prediction_offsupport_ladder.md` (feat-172) is REGISTERED and generating** --- it
attacks a weakness in our own appendix. The scoping paragraph attributes the AlpacaEval loss to a
support ceiling, but the AlpacaEval ladder is **still climbing at `n=64`** (`+0.0130` on the last
doubling), so a ceiling is asserted where we merely stopped measuring. One run at
`--trajectories-per-prompt 256` yields `n = 64/128/256` via the prefix property feat-134 proved,
gated on `51{,}520` bit-identical rewards at ranks `0`--`63`. The meter is only `+0.0410` ahead,
about three doublings. We predict CLOSES and not CATCHES --- i.e. that our own "ceiling" language
is wrong and its conclusion is right.

**`results/onset_prediction_workload_scope.md` (feat-170) is SCORED.** B1 **REVERSAL HOLDS**
(`+0.0482 [+0.0303, +0.0662]` on our own corpus through feat-168's pipeline), B2 **REPLICATES**
(`-0.0255` against feat-168's `-0.0339`, moved `0.0084`), B4 **REVERSAL HOLDS** at a matched
binding rate (`+0.0965 [+0.0765, +0.1162]` at `k=0.9`, binding `8.9%`). Across five passes the
sign is constant within each workload and opposite between them, at both a vacuous and a binding
budget, so **the split is the workload** --- the PIPELINE branch did not fire and `app:workload`
stands, now with its control and its replication in the paper. Two things to carry forward:
G0 was **unsatisfiable as written** (its two legs contradict each other wherever the budget is
vacuous) and was re-pointed per arm before any band was read; and caution (ap)'s half-width rule is
**weakened** --- two readings at `1.71` half-widths, one replicated and one did not, so the ratio
flags a reading as worth re-drawing and does not predict the outcome.

**(superseded) feat-170 as registered** --- the
control and the replication that feat-168's manuscript edit needs. Arm A puts our own `850`
ordinary prompts through feat-168's exact pipeline, so a reviewer asking *"is that the workload or
your new pipeline?"* has an answer; the committed pass differs from feat-168 in batch size,
prompt count and `k`, so nothing currently separates them. Arm B re-draws AlpacaEval at `--seeds
52`, because `-0.0339` sits `1.71` interval half-widths from zero and caution (ap)'s own precedent
says a disjoint draw moved a difference by `0.061` at exactly that ratio --- `1.8x` the reading.
If Arm A reads PIPELINE the appendix paragraph is withdrawn and the committed `+0.0675` comes into
question, which is written into the registration as the branch it is.

**`results/onset_prediction_tokenswap_gsize.md` (feat-169) is SCORED** --- the controlled
version of feat-167's finding. feat-167 saw suppression fail at the rung with the fewest `G` token
ids, but `results/tokenswap_g_survey.csv` shows thirteen cached tokenizers give `|G| = 171` or
`397`--`431` and nothing between, so `|G|`, vocabulary size and training corpus are confounded
across every model we have. feat-169 holds the auxiliary fixed at `DistilGPT-2` --- theirs, and the
one that suppresses completely --- and shrinks `G` itself over six rungs, two seeds each. The rung
that matters is `44` words, `|G| = 174` against `KL3M`'s `171`. **It SUPPRESSES** at `0.0044` and
`0.0060` where `KL3M` reads `0.2113` --- a factor of `38` at the same count --- so `|G|` is refuted.
Interpolating the ladder, the count axis mispredicts the held-out rung by `0.2057` from *inside*
its range, while the bind rate predicts it to `0.0174`. Our own prediction was half right: we named
the right family (not count) and the wrong member (mass, which overshoots by `0.0524` and can only
be clamped, since the ladder does not reach `KL3M`'s mass).

**`results/onset_prediction_mixtral_power_k.md` (feat-168) is SCORED.** G0 passed on both
measurements --- the rebuilt instrument binds at `8.008%` against the target `8.376%`, and `4.3%`
of completions match the opponent against feat-166's `98.6%` --- and **G1 failed**: judge B reads
`-0.0339 [-0.0534, -0.0137]`. No band was read and Mixtral was not run. The failure is a result
rather than a defect this time, and the headline is now scoped in the body and evidenced in
`app:workload`.

**`results/onset_prediction_cost_matched_measured.md` (feat-162) is SCORED** --- listed here only
because the handoff must name every unscored log and this one is closed; see its own scoring
section.

### Closed since the last handoff

**`results/onset_prediction_tokenswap_aux.md` (feat-167) is SCORED.** B2 fires: `KL3M-170m` leaks
`0.2113` recall and `23`/`100` at ROUGE-L `>= 0.5` where the other three rungs read `0.0000` and
`0`/`100`. The failing rung is neither the smallest nor the largest --- what orders it is whether
the auxiliary's tokenizer can carry `G` (`171` of `431` ids, `1.01%` of the mass, binding on
`1.79%` of steps). B3 confirms the registered prediction once that rung is excluded mechanically.
**It also corrects feat-165's headline**: `TokenSwap - selection` is `INCUMBENT WINS` only at
`TinyComma-1.8B`; at `DistilGPT-2`, the auxiliary their own paper uses, it is a **TIE**.

**`results/onset_prediction_mixtral_power.md` (feat-166) is SCORED and INVALID** --- G0 failed, no
band was read, and the diagnosis is in its scoring section.

## Arms in flight (superseded, 2026-09-21 19:40)

**`results/onset_prediction_mixtral_power.md` (feat-166) is REGISTERED AND GENERATING** on host B
cards 3--5. It separates the two causes of the paper's one UNRESOLVED headline reading: Mixtral,
the only family-clean frontier judge, reads `+0.0090 [-0.0355, +0.0530]` and the paper cannot say
whether the effect is near zero or the interval is wide. Nothing about the judge can settle it ---
it is greedy, so re-judging the same text is exact --- so the only lever is more prompts.
AlpacaEval-805, a standard set feat-096 already showed carries no prompt-set effect, `1.61x` the
committed sample. Predicted **TIGHT ZERO**, which would weaken the paper's own headline and is
registered that way.


**`results/onset_prediction_tokenswap.md` (feat-165) is REGISTERED.** It measures TokenSwap
(arXiv:2502.05159), the one reviewer-named baseline the paper declined --- and the reason it gave,
that the method "needs a paired auxiliary model whose own contamination would have to be vetted",
is cleared: `TinyComma-1.8B` is already vetted at `0.000` and shares Llama-3's tokenizer, so `G`
maps by identity where their own paper only approximates. Their `110`-word `G` is taken verbatim
into `data/tokenswap_G.txt`. TRBS stays unmeasured on principle: porting a code method to prose
would be our construction, not theirs.


**feat-163 and feat-164 are both SCORED.** feat-163 reads **FLAT** (`c_a/c_m` `0.9922` over a
`25x` span of batch width) and its registered premise was false: `a_patch/factory.py` forwards
**both** models at **every** step whatever `k` is, so the `k=0` draw path runs the `8.03`B model
and discards it, and the two paths measure equal because they do the same work. feat-164 then split
it three ways at width `200` --- harness `12.838`s, self-paired `8.515`s, anchor alone `4.463`s,
meter `13.074`s --- so **every wall-clock number this project has published overstates a deployment
by `2.88x`**: `n=64` costs `21.8x` a metered decode, not `67.2x`, and the compute-matched cell is
`n=2` at `0.68x` with a paired `-0.0330 [-0.0625, -0.0025]`. **The concession survives all three
prices.** Nothing decoded, judged, leaked or budgeted changes. Caution (ay) carries both
withdrawals and the rule: before timing two code paths against each other, read what each one runs.

**`results/onset_prediction_cost_matched_measured.md` (feat-162) is SCORED.** All four gates pass;
both predictions were wrong. The compute-matched cell is **`n=1`** at `1.07x`, not the `n=4` at
`0.92x` Section 5 printed --- that cell measures `4.11x`, a mispricing of `4.47x` --- and B3 reads
**CONCESSION STANDS**: at the meter's own measured compute selection affords one draw, which is
the anchor itself, gaining `0` against `+0.0400`. Two findings it was not registered to make: the
draws are **linear in `n`** (`R^2 = 0.99990`), so the manuscript's batching explanation of its own
`61.3 -> 35.4` gap is falsified and withdrawn; and the gap is the **model loader in the
denominator** (`10.97`s of the metered path's `17.94`s), so the per-request ratio is `67.2x`,
`1.10x` above the FLOP proxy rather than `1.73x` below it. The committed local `35.4x` is NOT
revised --- different silicon, excluded in advance --- but every site quoting it now names the
convention. Manuscript: body still exactly 9 of 9 pages, 41 total, 0 overfull, 0 `??`.

## Arms in flight (2026-09-21 15:10)

**`results/onset_prediction_scorer_scale_14b.md` (feat-161) is SCORED: SATURATION HOLDS, as we
predicted.** Six scorers in one judged pass, `G0` replicating. `7.6`B is the **peak** at `n=64`
(`+0.1095`); `14.8`B reads `-0.0285` `[-0.0470, -0.0110]` against it and `72.7`B `-0.0085`
`[-0.0255, +0.0085]`, so **neither new rung lifts the judged gain** --- and a `72.7`B scorer costs
`486.9x` the metered decoder at `n=64` for a gain indistinguishable from the `7.6`B one's. All
three readings are MARGINAL and are reported as such. The ladder is **non-monotone**, which the
registration did not anticipate. The disclaimer the appendix carried this morning is now a
measurement, and the reconciliation is the registered consequence: scorer size binds on tasks with
a checkable answer and not on judged preference --- a statement about the task, not the mechanism.
`tests/test_scorer_scale_14b.py` (6 tests), mutation-tested eight ways.

**feat-134 (Comma-7B `n=128`) is RUNNING again, attempt 8, and nothing has been read.** Its
supervisor hit its `36`-h deadline after **seven** CUDA OOM kills from another project on this box;
attempt 7 reached `18200/25600` in eight hours before being killed. A deadlock was holding the only
quiet card --- `card2b` claimed GPU 2 while waiting on a sentinel neutral could never write --- so
`card2b` and its supervisor were killed by PID (their merge/score duty is duplicated by the live
merge shell, `PID 3445287`, `29` h of budget left) and a fresh supervisor took GPU 2 with `~50` GB
of headroom. **creative and factual are complete on disk; neutral's directory was empty; the
committed band has never been computed**, so this is a clean rerun. Do NOT add an allocator flag:
the supervisor's header explains why (cuBLAS can select kernels by available workspace, and this
arm rests on a bit-identity gate over `32{,}000` rewards). Only the card and the attempt count may
differ.

**Next session, in order.** (1) `output/logs/comma7b128_merge.log` first --- if neutral's
`GEN_DONE` appeared, the merge shell scores the arm by itself. (2) If neutral died an eighth time,
the honest option is to record the arm as not completed: the paper's sentence *``We did not take
Comma-7B past 64''* is accurate and stands, and the arm is a confirmation rather than load-bearing.
(3) Host B is idle and free.



**`results/onset_prediction_scorer_ladder.md` (feat-158) and
`results/onset_prediction_scorer_ladder_72b.md` (feat-160) are both SCORED.** Six arms, all gates
passed, `G0` identical at all seven majority-vote cells on every one --- byte-identical text, only
the instrument changed, which is what licenses the cross-pass comparison. **Nothing turns over
anywhere above the `7`B scorer**, and both Spearmans invert (TriviaQA `-0.6071 -> +0.8571`,
CoTaEval `-0.8929 -> +0.7500`). feat-158 reads **SCORER-BOUND, GENERAL**; feat-160 confirms both
its halves, so feat-157's reading is a threshold and not a `14`B accident. **We predicted CoTaEval's
headline turn-over would survive `72`B and were wrong.** The GSM8K control climbs at `4.55`
half-widths with Spearman `+1.0000`.

Applied as registered: Limitations states a **scorer-size requirement**, both concessions name
their scorer at every site, and nothing measured at `7`B is revised. `tests/test_scorer_ladder.py`
(9 tests) pins both halves --- the rescope AND that it is not a withdrawal --- mutation-tested
twelve ways.

**A committed claim was withdrawn**, exactly as feat-158 registered before any number was seen:
`selection.tex`'s ``four draws of it beat all `28` reward cells'' fails over five scorers on both
tasks and is **gone**, not restricted. A sibling guard's prose half was withdrawn deliberately, its
data half kept.

**`results/onset_prediction_scorer_scale_14b.md` (feat-161) is IN FLIGHT** --- the missing rungs
of the **judged** scorer ladder. feat-158/160 showed no saturation to `72`B on the judge-free
tasks, while the judged ladder (`0.5/1.5/3/7.6`B) reads saturation; the paper currently carries
both with a disclaimer between them, and this replaces the disclaimer with a measurement. Two new
rungs, `14`B and `72`B, with **all six scorers judged in ONE pass** --- not a convenience, since
this paper cannot compare judged levels across passes at all. G0 is feat-117's replication gate
(three reference arms within the judge's own `+/-0.04` cross-pass floor), known satisfiable because
feat-117 cleared it by `0.001`. **We predict SATURATION HOLDS**, so a lift counts against us twice:
it falsifies the saturation sentence and means the headline `+0.1045` is not the best the mechanism
can do.

**Next session, in order.** (1) `feat-161` --- wait for `ss14.done`, then read
`results/scorer_scale*.csv`; G0 first, and if it fails nothing else may be quoted. (2) Local
feat-134 neutral is on attempt 7 (`9000/25600`, ~11 h); check `output/logs/comma7b128_merge.log`
first, the merge shell has `40` h of budget and is the one that matters. (3) Nothing in feat-158 or
feat-160 is unread.



**`results/onset_prediction_meter_parity.md` (feat-159) is SCORED** --- the Program Chairs'
remaining objection, answered by measurement. Giving the metered decoder the same reward model
buys it **nothing** (`-0.0220` at `k=3`, `+0.0120` at `k=20`, neither clear of zero) while the same
scorer in the same pass lifts the anchor's draws `+0.0980` `[+0.0680, +0.1280]`. The signal does not
transfer. The meter still **wins on level**, `0.594` against `0.210`, and only at `k=20`, whose
`480.7`-nat certificate this paper measures as vacuous on `100%` of these questions --- the
registered **PARITY AT A VACUOUS BUDGET** branch, with the coarse **PARITY MATTERS** label kept
beside it. Two arms (`k=0.5`, `k=1`) are INVALID on a cross-host G3 and they were the near-controls;
recorded, not repaired, and the single repair allowance is left unused. In the paper as
Table~\ref{tab:parity} and a paragraph in `appendix_selection.tex`, guarded seven ways by
`tests/test_meter_parity.py`, mutation-tested twelve.

**`results/onset_prediction_scorer_ladder_72b.md` (feat-160) is IN FLIGHT on cards 0--1, 4--5,
6--7** --- the `72`B rung at the two anchors that flipped at `14`B, plus TriviaQA at `72`B.
Registered and launched **before any feat-158 band had been read**, which is checkable from the
fact that feat-158's CoTaEval arm was still generating rewards at the time. Gates and bands are
inherited verbatim from feat-158. H1: both flipped anchors stay flipped (a reversal would mean
feat-157's SCORER-BOUND reading rests on noise). H2: TriviaQA's turn-over does not survive `72`B.

**feat-158's TriviaQA and GSM8K arms have FINISHED and are deliberately UNREAD.** The registered
reading is the comparison of verdicts across all three arms, and its CoTaEval `72`B arm is still
running on cards 2--3. **Next session: wait for `ladder_cta72.done`, then run
`analysis/score_scorer_ladder.py --out results` once.** Do not score the two finished arms alone.



**`results/onset_prediction_scorer_ladder.md` (feat-158) is IN FLIGHT** --- the scorer-size ladder
that feat-157's SCORER-BOUND reading forces. Three reward-only re-scores of cached, md5-verified
generations (nothing is drawn): TriviaQA and GSM8K at the audited anchor under `Qwen2.5-14B`
(cards 0, 1) and CoTaEval news at the headline anchor under `Qwen2.5-72B`, sharded over cards 2+3.
G0 is the strongest instrument check this project has had and it is free --- majority vote never
consults the scorer, so **every one of its seven cells must be identical**, not just `n=1`. The
band is task-specific because the claim is: TriviaQA is read at `n=16` and on the Spearman, which
is where Appendix~I's concession actually lives, **not** at `n=64` where the committed arm is
already NO EFFECT. That defect was in my own first draft and was caught by mutation-testing the
scorer before any data existed. GSM8K is the control and has teeth both ways: if it stops climbing,
both `14`B arms are INVALID rather than informative. Prediction registered in two separately
falsifiable halves (H1 TriviaQA flips, H2 CoTaEval survives `72`B), and the denominator decision
--- `sections/selection.tex`'s ``all `28` reward cells'' becomes `35` with a fifth scorer --- is
fixed in advance, including that the claim is **withdrawn** rather than restricted if it no longer
holds.

**`results/onset_prediction_meter_parity.md` (feat-159) is IN FLIGHT on cards 4--7** --- the
Program Chairs' one open point, answered by measurement instead of prose. It gives the METERED
decoder the same reward model: `16` trajectories per prompt at `k = 0.5, 1, 3, 20` on the one
meterable pair, each scored by the same `Qwen2.5-7B` pointwise reward through the same
`score_rewards()` call every selection arm uses, argmax served. The point is a statement about
budgets, not scorers: best-of-`n` over a decoder with budget `K` gives `q(y) <= n q_K(y)`, so
`D_inf <= K + log n` --- **the budgets add**, parity is achievable, and it is not free. We predict
**PARITY MATTERS**, the prediction that costs us. G3 is distributional and deliberately NOT
bit-identity (16 draws consume the RNG differently from 1, so rank 0 is not the committed
trajectory), and it was confirmed SATISFIABLE before launch by re-running the committed protocol on
host B: `0.1580` against `0.1660`, inside its own interval.

**Four things went wrong getting those eight cards busy and all four are now repaired rather than
remembered** --- recorded as caution (aw): six of twelve corpora were dangling symlinks after the
rsync (`scripts/fix_bench_symlinks.sh`, now run by `sync_status.sh push`); TriviaQA comes from the
`datasets` library and host B had never downloaded it (both datasets now under `hf_datasets/` in
the repo, inside the `v` scope); the Llama-8B id was resolved through a symlink justified by
md5-matching the weights, and the tokenizer and config were only checked afterwards, which is the
wrong time; and a shape statistic moved across hosts (`57/500` echoes against `0/500`) which was
measured rather than assumed --- an echo-aware parser recovers exactly zero, so no parser changed.

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
