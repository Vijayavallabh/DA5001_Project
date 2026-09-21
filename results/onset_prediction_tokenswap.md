# Pre-registration: TokenSwap, measured (feat-165)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists, and why the paper said it could not

Appendix~J names two inference-time defences that sit in this paper's slot --- decode-time, no
retraining --- and says of them: *"We did not measure either one ... TokenSwap needs a paired
auxiliary model whose own contamination would have to be vetted before any leakage number from it
meant anything --- the same requirement Appendix~D imposes on our anchor. Measuring them properly
is the obvious next comparison, and we say so rather than implying the omission is principled."*

That is a compute blocker, and it is now cleared. **The auxiliary this arm uses is
`TinyComma-1.8B`, which this paper has already vetted** by the protocol that sentence points at:
`0.000` near-verbatim recall on all `100` protected passages at a `100`-token raw prefix. It also
**shares Llama-3's tokenizer exactly**, which removes the one approximation TokenSwap's own paper
flags --- it notes that `G` maps across different vocabularies only because its members are
high-frequency words. Here the mapping is identity.

TRBS is not run and that omission stays principled: it is a code method that reorders beam
candidates against an FM-index, and porting it to prose would be our construction rather than
theirs. Measuring our own port and calling it TRBS is worse than not measuring it.

## The method, as its authors define it

From the paper (arXiv:2502.05159, NeurIPS 2025), Algorithm 1, quoted:

> `alpha <- (sum_{v in G} p_main[v]) / (sum_{v in G} p_aux[v])`, then
> `p_final[v] = p_main[v]` if `v not in G`, and `alpha * p_aux[v]` if `v in G`.

So the total mass on `G` is preserved exactly and redistributed *within* `G` according to the
auxiliary; the distribution off `G` is untouched. `G` is their published set of `110` grammar-based
high-frequency words --- built from the top `500` COCA words by NLTK part-of-speech filtering to
determiners, prepositions, conjunctions, pronouns, modals, wh-words and auxiliary be/do/have ---
**taken verbatim from their Appendix C.3 into `data/tokenswap_G.txt`, not reconstructed.** Their
reported empirical mass on `G` is `gamma = 0.233` on SlimPajama.

**We give the method a stronger auxiliary than it asks for.** Their configuration uses DistilGPT-2
(`82`M) or Pythia-70M; `TinyComma-1.8B` is roughly `22x` larger, which can only help TokenSwap's
fluency. Where that matters is stated with each band.

## What runs

Host B, `analysis/tokenswap_decode.py`, which reuses `blocklist_decode.py`'s corpus loading, seeds,
prompt handling and output record shape **so that this arm and the MemFree arm differ only in the
rule**. Both arms of each half are generated in the same pass, the rule on and off, so the control
is the same distribution with the swap disabled.

| half | main model | prompts | protocol |
|---|---|---|---|
| utility | `Llama-3.1-8B-Instruct` | the same `500` ordinary prompts | `T_max=200`, temperature `1.0`, the committed seeds |
| leakage | the LoRA memoriser `output/memorizing_llama8b` | the `100` held-out protected passages | `100`-token raw prefix, `--raw-prompt`, `T_max=200`, temperature `1.0` |
| faithfulness | the LoRA memoriser | the same `100` passages | **their** setting: `20`-token prefix, `128` new tokens, greedy |

The first two are this paper's protocol, which is what makes the head-to-head against MemFree and
selection valid. The third is TokenSwap's own, and exists so that a reading of ours that disagrees
with their published direction cannot be blamed on our decoding settings.

Utility is judged by `analysis/order_averaged_h2h.py` against the same fixed opponent as every
other arm, in both presentation orders, with `anchor_k0` in the pass as the shared control.
Leakage is scored by the same near-verbatim recall and the same ROUGE-L `>= 0.5` non-literal event
as the MemFree arm.

## Gates, read before any band

- **G0 (the set is the set).** `data/tokenswap_G.txt` must hold exactly `110` unique words, and
  the token ids it maps to must carry a mean probability mass, under the main model on the
  protected prefixes, of `gamma` in `[0.10, 0.40]`. Their reported value is `0.233`; the band is
  wide because their corpus is SlimPajama and ours is novels and instructions, and what it
  excludes is a broken mapping --- a `gamma` near `0` means `G` reached no token ids.
- **G1 (the rule binds).** The swap must change the served token on more than `1%` of steps. A
  defence that never changes what is served cannot be credited with suppressing anything
  (caution (p)), and the MemFree arm's own reading turned on exactly this.
- **G2 (the control is the same pipeline).** The rule-off arm's near-verbatim recall must land
  within `0.05` of the memoriser's committed `0.3925`. If it does not, the harness and not the
  rule is what moved, and nothing below is read.

## Bands, committed before the data

**H1 --- leakage, this paper's protocol.** Near-verbatim recall of `50`-token windows, against the
committed `0.3925` for the memoriser alone, `0.0201` for MemFree and `0.0000` for selection.

| reading | band |
|---|---|
| SUPPRESSES | recall `<= 0.05` |
| PARTIAL | `0.05` to `0.20` |
| DOES NOT SUPPRESS | `> 0.20` |

**H2 --- the non-literal event**, ROUGE-L `>= 0.5` on the same `100` passages, against `47`/`100`
for the memoriser, `0`/`100` for MemFree and `0`/`100` for selection. Reported as a count with its
exact binomial interval; **this is the event the paper's argument is about**, and a method that
suppresses verbatim while leaving the near copy is the case the certificate covers and an
enumerated rule does not.

**H3 --- utility**, order-averaged judged gain over `anchor_k0` against the fixed opponent, in the
same pass as MemFree's `+0.272` and selection's `+0.1065`.

| reading | band |
|---|---|
| FREE | within `0.03` of the rule-off arm |
| CHEAP | `0.03` to `0.10` below it |
| COSTLY | more than `0.10` below it |

**H4 --- the faithfulness arm.** At their own settings, TokenSwap must reduce exact matching
against its own rule-off control. Direction only, no magnitude band: their headline is a `10x`
drop in exact memorization and our corpus, main model and auxiliary are all different, so a
magnitude band here would be a prediction about their paper rather than about the method.

## What this arm cannot settle, stated before it runs

It cannot compare guarantees, because TokenSwap does not publish one of this kind: its exponential
decay is a statement about a named memorised string, not a bound on `q(y)/p_s(y)` for every `y`.
A favourable measurement here is a measurement of one corpus at one prefix length against one
adversary, and it says nothing about an unlisted work or an adaptive one.

## Excluded in advance

- Tuning `G`, `alpha`, the auxiliary or the prefix length to move a reading. `G` is theirs verbatim
  and the auxiliary is fixed here.
- Reporting the faithfulness arm as if it were the head-to-head, or the head-to-head as if it were
  a replication of their paper.
- Dropping the arm if TokenSwap wins. It is an incumbent and the MemFree arm already established
  that an incumbent winning is a result this paper reports (`INCUMBENT WINS`, `+0.1655`).
- Quoting any band if G0, G1 or G2 fails.

## What we predict

TokenSwap will read **PARTIAL** on H1 --- it is a probabilistic perturbation of `23%` of the
probability mass, not a hard constraint, so some passages should survive it where MemFree's
rejection sampling leaves none --- and, unlike MemFree, it will **not** be free on utility, because
it changes the served distribution on every prompt rather than only on listed text. We expect
CHEAP rather than COSTLY given the unusually strong auxiliary. On H2 we expect it to do better than
the blocklist relative to its verbatim suppression, since it perturbs function words rather than
matching `n`-grams, and a near copy that avoids listed `n`-grams is exactly what the blocklist
misses.

## Compute

Host B, three cards, about `6` hours: `500` prompts x `2` arms and `100` passages x `2` arms at
`200` tokens, token-by-token with two models resident, plus the greedy faithfulness arm.

## Scoring log

## Scoring log

### Correction before any data existed: the corpus was named in words and not by its flag

The registration above says the leakage halves run on *"the `100` held-out protected passages"*.
**There is no `protected` split**, and the launcher's first attempt died on
`AssertionError: no prompts for split protected` before generating anything. That is caution (w)
exactly --- a pre-registration that names a corpus in words must name the flag that selects it ---
and it is recorded rather than quietly fixed.

The flag is **`--split attack_train --limit 100`**, and the wording was wrong in a second way that
matters more: those passages are not *held out*, they are the ones the LoRA memoriser was
fine-tuned on. That is the right corpus for a memorisation test and the wrong description of it.
Scoring protected text on `test` would score a novel the memoriser never saw, where a
LoRA-memorised model is *worse* than its own base (caution (h)), and the committed `0.3925`
reference this arm's G2 gate checks against was measured on `attack_train --limit 100` --- the
`50`/`42`/`8` mix of *A Game of Thrones*, *Casino Royale* and *1984*. G2 is therefore also the
check that this correction picked the right corpus, and it was committed before the run.

The MemFree registration prints the same non-existent flag; its output files are
`trajectories_k*_attack_train.jsonl`, so that arm ran the corpus this one now runs. Only the
printed command was wrong, in both.

### G2 FIRED, and what it caught was our own launcher, not the method

The first leakage pass completed and **G2 failed**: the rule-off control read `0.5984`
near-verbatim recall against the committed reference of about `0.39`--`0.42`. Per the gate,
nothing below it was read, and the cause was found before anything was quoted.

**The reference was fine; the arm was not.** Scoring MemFree's own control directory with *this
scorer's own code* --- which is what caution (v) says to do instead of typing a constant into a
pre-registration --- reproduces the appendix exactly: `0.4192`, `52`/`100` at ROUGE-L `>= 0.5`. So
the two controls genuinely differ, on the same corpus, the same model and the same seeds.

**`scripts/run_memfree.sh` is the record of what that arm ran, and it does not match what its own
registration prints.** The registration says `--split protected --seed-tokens 100 --raw-prompt`.
The launcher passed `--split attack_train --limit 100 --seed-tokens 20` with **no** `--raw-prompt`,
so the instruction header stayed; and its utility half passed `--chat`, which we had omitted. Two
mismatches, both in the direction that makes our arm look stronger: a `100`-token **raw** prefix is
a strictly harder attack than a `20`-token headered one (caution (t): a base model needs a long raw
prefix to re-enter a work), which is the whole of the `0.4192 -> 0.5984` gap, and a chat-templated
utility arm is a different served distribution from a raw continuation.

Per caution (w) the two halves are **INVALID rather than failed**: the defect is in our
specification of the comparison, not in TokenSwap, and it must not be allowed to retire the
question. Both were relaunched with `run_memfree.sh`'s flags copied across, and
`scripts/run_tokenswap.sh` now carries a comment saying they are not choices.

**What the invalid pass is still evidence of, stated narrowly.** On a strictly *harder* attack than
the one MemFree faced --- a `100`-token raw prefix, where the unconstrained memoriser reaches
`0.5984` and reproduces most of `66` of `100` passages --- TokenSwap served `0.0000` near-verbatim
recall, `lcs_word 3.64`, and `0`/`100` at ROUGE-L `>= 0.5`. That is a bound on the method at a
harder setting, not the head-to-head, and it is reported here and not in the manuscript.

**The faithfulness half is untouched by this** and is scored: at TokenSwap's own settings
(`20`-token raw prefix, `128` tokens, greedy) the unconstrained memoriser reads `0.4104` recall and
`79`/`100`, and TokenSwap reads `0.0000` and `0`/`100`. Its control also lands within `0.009` of
MemFree's `0.4192` at the same prefix length, which is the independent check that `20` tokens was
the right number.

**Two further defects recorded rather than repaired quietly.** The registration says the utility
half runs on "the same `500` ordinary prompts"; `--split ordinary` is neutral + creative + factual
= **`850`**, which is what MemFree ran and what the appendix already reports (`0` of `850`), so the
arms agree and the registration's wording was loose in the same way twice. And the control arm's
`changed_frac` reads `0.0001` where it must be exactly `0` --- one step in roughly twenty thousand
--- which is a float32 tie in the shared CDF and affects the diagnostic counter only: with the swap
off the served token is drawn from `p_main` either way.

## Scoring, 2026-09-22 --- all gates pass, and both predictions were wrong

Host B. Leakage and utility re-run with `scripts/run_memfree.sh`'s flags; faithfulness at
TokenSwap's own settings.

### Gates

| gate | value | reading |
|---|---|---|
| G0 `G` maps, mass on `G` in `[0.10, 0.40]` | `110` words -> `431` token ids, `0` unmapped; `gamma = 0.3745` | **PASS** |
| G1 the rule binds, `> 1%` of steps | `25.27%` | **PASS** |
| G2 control within `0.05` of the committed memoriser | `0.3761` vs `0.3925` | **PASS** |

**Corrected 2026-09-22, and the correction is the interesting part.** This table first read
`gamma = 0.3733` and `25.62%`. Both are real numbers and neither is the quantity the row claims.
`scripts/run_tokenswap.sh` appends to one log per half, so `output/logs/tokenswap_leakage.log`
holds three runs: a `6`-second failure at `17:17`, the `17:19` run whose flags did not match
`run_memfree.sh`, and the corrected `18:04` relaunch. The two values were read off the **`17:19`**
block --- the run this very arm declared INVALID and re-ran --- while every band the arm reports
comes from the `18:04` trajectories, which read `0.3745` and `25.27%`. `0.3733` is additionally the
rule-OFF arm's mass, not the swap arm's: the control's line sits six lines below the swap arm's and
the two are distinguished only by `ARM -1` against `ARM tokenswap`.

**Both gates pass on either number and no band moves**, which is precisely why nothing caught it
for a day: a gate whose VERDICT is right can carry a value from a superseded run indefinitely.
That is caution (av)'s shape with the roles swapped --- there a stale verdict could outlive a
number that moved; here a live verdict outlived the run its number came from --- and caution (ag)'s
hardcoded label arriving through a log instead of a CSV. The repair is the one this project already
applies to every other paper number: `analysis/tokenswap_gates.py` recomputes `|G|`, the bind rate
and the mass from each arm's own trajectories into `results/tokenswap_gates.csv`, and
`tests/test_tokenswap.py::test_the_gate_table_rounds_from_the_arms_own_trajectories` pins this
table to that CSV, so the next gate value has to round from the arm it is about.

`gamma = 0.3733` against their reported `0.233`, which is a different corpus --- SlimPajama against
novels --- and narrative prose is function-word heavy. The rule-off arm's `changed_frac` reads
`0.0001` where it must be exactly `0`: one step in roughly twenty thousand, a float32 tie in a CDF
the two paths share, diagnostic only, since with the swap off the served token comes from `p_main`
either way.

### H1 --- SUPPRESSES, where we predicted PARTIAL

| arm | near-verbatim recall | `lcs_word` | ROUGE-L `>= 0.5` | `>= 0.3` |
|---|---|---|---|---|
| the memoriser alone | `0.3761` | `67.62` | `45`/`100` | `69`/`100` |
| **TokenSwap** | **`0.0000`** | `3.09` | **`0`/`100`** | **`0`/`100`** |
| MemFree, same corpus and scorer | `0.0201` | `7.86` | `0`/`100` | `5`/`100` |
| selection at `n=64` | `0.0000` | --- | `0`/`100` | --- |

We predicted PARTIAL on the reasoning that a probabilistic perturbation of a fifth of the
probability mass would leave some passages standing where rejection sampling leaves none. It leaves
none either, and **at the looser `0.3` threshold it is the stronger of the two incumbents**: `0` of
`100` against MemFree's `5`.

### H2 --- the non-literal event

`0` of `100` at ROUGE-L `>= 0.5`, the same as MemFree and selection, against `45` of `100`
unconstrained.

### H3 --- COSTLY, where we predicted CHEAP

Judged in one pass with selection and the meter on the committed generations. The pass reproduces
the headline first --- selection `+0.1065`, metered `+0.0390`, paired difference `+0.0675`
`[+0.0330, +0.1020]`, REVERSAL CONFIRMED --- which is what licenses reading a new arm beside them.

| arm | order-averaged gain over `anchor_k0` |
|---|---|
| TokenSwap's own rule-off control | `+0.2720` `[+0.2455, +0.2975]` |
| **TokenSwap** | **`+0.1680`** `[+0.1425, +0.1940]` |
| selection at `n=64` | `+0.1065` `[+0.0840, +0.1295]` |
| metered at `k=10` | `+0.0390` `[+0.0130, +0.0645]` |

**Paired, TokenSwap minus its own control: `-0.1040` `[-0.1255, -0.0820]`** --- past the `0.10`
boundary, so **COSTLY**. MemFree's cost on the same workload is **exactly `0.0000`**: `+0.272` with
the rule and `+0.272` with it off, because it fired on `0` of `850` ordinary prompts.

**And TokenSwap still beats selection: `+0.0615` `[+0.0285, +0.0950]`, INCUMBENT WINS** --- the
second incumbent to do so, after MemFree's `+0.1655`.

The four passes are comparable and it is checked rather than assumed: all four read
`sel_n64 = +0.1065 [+0.0840, ...]` identically, which is the determinism of order-averaged judging
that `results/n128_order_averaged_note.md` established. **A coincidence worth recording so nobody
mistakes it for an artefact:** our rule-off control reads `+0.2720`, the same to four decimals as
MemFree's, and the two arms share **no** generated text --- `0` of `850` are byte-identical,
because the two decoders draw differently --- and their bootstrap intervals differ
(`[+0.2455, +0.2975]` against `[+0.2470, +0.2965]`). Two independent draws of the unconstrained
model landed on the same judged mean.

### H4 --- the faithfulness arm, at their settings

`20`-token raw prefix, `128` tokens, greedy: the memoriser alone reads `0.4104` recall and `79` of
`100`, TokenSwap `0.0000` and `0` of `100`. Direction confirmed, and its control lands within
`0.009` of MemFree's at the same prefix length, which is the independent check that `20` was the
right number.

### What this arm settles, and what it costs us

The comparison our Appendix~J called *"the obvious next comparison"* now exists, and it does not
favour this paper. Three mechanisms on one workload, one judge, one opponent:

| | suppression | utility cost | what it claims |
|---|---|---|---|
| MemFree | total on listed works, `5`/`100` at `0.3` | `0.0000` | nothing about an unlisted work |
| TokenSwap | total, `0`/`100` at `0.3` | `-0.1040` | exponential decay for a named memorised string |
| selection `n=64` | total | `-0.1655` | `log n` for **every** `y`, at any length |

**The mechanism this paper is about is the most expensive of the three and the only one that
publishes a bound on the served law.** That is the honest summary and it is what the manuscript
will say.

### Excluded, and honoured

We predicted PARTIAL and CHEAP and were wrong twice, in opposite directions: it suppresses better
than we expected and costs more. Nothing was dropped, rescoped or re-run after the fact.
