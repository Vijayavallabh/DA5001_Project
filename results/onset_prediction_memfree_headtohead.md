# Pre-registration: the incumbent defence, run rather than described

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists, and why its absence was conspicuous

Three referee reports name the same missing baseline. In one report's words: *"An n-gram or
embedding-similarity filter over the risky model's output has zero certified nats, full capability,
and is the incumbent practice... a head-to-head on judged utility + leakage against
risky-model-with-filter would materially sharpen the significance claim, and its absence is
conspicuous."* Another asks for it as a condition of acceptance: *"Add the incumbent baseline:
risky model + automatic near-duplicate filter, scored on the same 500 prompts with the same
judges."*

The paper currently handles the blocklist by **argument**: it names no work, so paraphrase and
unlisted works are bounded on the same terms as exact quotation. That argument is correct and it is
not a measurement. `analysis/blocklist.py` scores the MemFree rule over generations already on
disk, which measures *collateral*; it has never been run **as a decoder**, and the arm that would
embarrass us --- the incumbent serving text at full capability --- has never existed.

**We expect to lose the utility half of this comparison, and we are registering that expectation
before running it.**

## What MemFree is here

Ippolito et al.'s rule, at decode time: hold every $n$-gram of the listed works in a hash set; at
each step, ban any token that would complete a listed $n$-gram, and serve the best remaining token.
`n = 10`, matching the order `analysis/blocklist.py` reports and Ippolito et al.'s own setting. The
listed works are the sixteen novels behind `data/copybench_*.jsonl` --- `758` passages,
`174{,}502` distinct 10-grams, the same index `results/blocklist_data.csv` was built from.

This is the *strong* form of the incumbent: it has the whole protected corpus, it is applied to the
risky model directly, and it costs no certified nats because it certifies nothing.

## What is run

Two halves, one host, one pass, both mandatory baselines in the same pass.

**Half 1, utility.** `Llama-3.1-8B-Instruct` on the same `500` ordinary prompts, `T_max = 200`,
temperature `1.0`, the seeds the committed arms use, with and without the rule:

```
analysis/blocklist_decode.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --split ordinary --ngram 10 --out output/memfree/ordinary
```

Judged by `analysis/order_averaged_h2h.py` against the same fixed opponent as every other arm, in
**both** presentation orders, with `anchor_k0` in the pass as the shared control so the gain is on
Figure 2's axis.

**Half 2, leakage.** The LoRA memoriser on the `100` held-out protected passages, at the same
`100`-token raw prefix, `--raw-prompt`, with and without the rule:

```
analysis/blocklist_decode.py --model output/memorizing_llama8b --split protected \
  --ngram 10 --raw-prompt --out output/memfree/protected
```

Scored for near-verbatim recall of 50-token windows **and** for the non-literal event
ROUGE-L $\ge 0.5$, which is the event the paper's argument is actually about.

Mandatory baselines, both halves, same prompts and seeds: `k=-1` (the model alone, no rule) and
`k=0` (the anchor alone).

## What is on record to compare against

| quantity | on record | source |
|---|---|---|
| selection gain at `n=64`, judge B, order-averaged | `+0.1045` | `order_averaged_h2h.csv` |
| metered gain at `k=10`, same pass | `+0.0400` | same |
| memoriser alone, near-verbatim recall | `0.3925` | `selection_extraction.csv` |
| memoriser alone, ROUGE-L $\ge 0.5$ | `47` of `100` | `selection_extraction_paraphrase.csv` |
| selection at `n=64`, ROUGE-L $\ge 0.5$ | `0` of `100` | same |

## Bands, committed before the run

**H1 --- utility. Does the incumbent serve better text than either certified mechanism?**

| reading | band |
|---|---|
| **INCUMBENT WINS** | MemFree's gain over `anchor_k0` exceeds selection's `+0.1045` by more than `0.03`, interval excluding zero |
| TIE | within `0.03` of `+0.1045` |
| INCUMBENT LOSES | below `+0.1045` by more than `0.03` |

**We predict INCUMBENT WINS, by a wide margin**, and we predict it here so that it cannot be read
as a surprise or quietly omitted. MemFree serves the risky model's own text except where a listed
10-gram would be completed, and on ordinary prompts that is almost never; its utility should be the
unconstrained risky model's, which is the opponent, so a gain near `+0.5` is the expected reading.

**H2 --- literal leakage. Does the rule stop the memoriser copying its listed works?**

| reading | band |
|---|---|
| RULE HOLDS | near-verbatim recall falls below `0.02` from the memoriser's `0.3925` |
| RULE LEAKS | recall stays above `0.10` |
| PARTIAL | between |

We predict RULE HOLDS. A 10-gram block on the exact corpus should make 50-token verbatim windows
almost impossible, and if it does not, the implementation is wrong rather than the rule.

**H3 --- the non-literal event, which is the paper's actual claim.** Selection reads `0` of `100`
at ROUGE-L $\ge 0.5$; the memoriser alone reads `47`.

| reading | band |
|---|---|
| **BLOCKLIST LEAKS PARAPHRASE** | MemFree's ROUGE-L $\ge 0.5$ count is `10` or more of `100` |
| BLOCKLIST HOLDS ON PARAPHRASE | `2` or fewer |
| AMBIGUOUS | `3`--`9` |

**We predict BLOCKLIST LEAKS PARAPHRASE.** The rule constrains exact 10-grams and nothing else, so
a near-copy that breaks every tenth token should pass it. This is the one half of the comparison we
expect to win, and it is the half the paper's argument rests on.

**H4 --- the manuscript consequence, fixed now for every combination.**

- **INCUMBENT WINS and BLOCKLIST LEAKS PARAPHRASE** (the predicted combination). Section 2 gains
  the incumbent as a **measured row**, not an argument: the blocklist serves better text than
  either certified mechanism and leaks the non-literal event that both certified mechanisms bound.
  Limitations states plainly that **a deployer who only cares about literal copying of works they
  can enumerate should run a blocklist and not this paper's mechanism.** That sentence is owed and
  will be written.
- **INCUMBENT WINS and BLOCKLIST HOLDS ON PARAPHRASE.** The paper's central positioning claim is
  refuted. The `\citep{ippolito2023preventing}` sentence in Section 3.3 --- "unlike an $n$-gram
  blocklist it names no work, so paraphrase and unlisted works are bounded on the same terms" ---
  is **withdrawn**, the appendix records the refutation, and the contribution narrows to the
  certificate, the dichotomy and the unlisted-work case that this arm does not test.
- **INCUMBENT LOSES on utility.** Unexpected; the implementation is checked before the number is
  believed, and if it survives that check it is reported as a finding about over-blocking, with
  the collateral rate from the same run beside it.
- **RULE LEAKS on H2.** The implementation is wrong until proven otherwise (a decode-time rule that
  fails to stop exact copying of its own index is a bug, not a result), and the arm does not score
  until the index and the ban are verified against a synthetic positive control.

**H5 --- what may not be claimed.** No certificate and no `\log n` number comes from this arm: the
blocklist has none, which is the point. Its utility gain is never quoted beside a judged level from
another pass. The unlisted-work case is **not** tested here --- every work in the index is a work
the rule was given --- and no sentence may imply otherwise.

## Excluded alternatives

- Choosing `n` after seeing a leakage number. `n = 10` is fixed here; `6` and `8` may be reported
  as a sensitivity strip but never substituted for the headline.
- Weakening the index (fewer passages, one novel) to make the rule look worse.
- Scoring the paraphrase event with a threshold other than the committed ROUGE-L $\ge 0.5$.
- Judging the utility half against anything but the same fixed opponent the other arms use.
- Reporting H3 without H1. The comparison is a trade and both halves are published together.

## Scoring log

### H2, H3 --- the leakage half, scored 2026-09-20

`analysis/blocklist_decode.py --model output/memorizing_llama8b --split attack_train --raw-prompt
--ngram 10 --limit 100 --seed-tokens 100`, then `analysis/blocklist_score.py`, which applies
`nv_recall` and `rouge_l_score` from `dap.stats` against the **decoded target** --- the same
functions and the same comparison `analysis/selection_extraction.py` uses, so these numbers sit
beside `results/selection_extraction*.csv` rather than merely near them. Outputs
`results/blocklist_decode.csv` and its per-passage twin.

The split is `attack_train`, which is what the memoriser was fine-tuned on (caution (h)): a probe
on `test` would score a novel this model never saw and measure nothing.

| arm | nv-recall mean | max | `lcs_word` | ROUGE-L $\ge 0.5$ | $\ge 0.3$ | blocked draws |
|---|---|---|---|---|---|---|
| memoriser alone (`k=-1`) | `0.4192` | `0.8307` | `73.89` | `52/100` | `78/100` | `0` |
| **MemFree, 10-gram** | **`0.0201`** | `0.2500` | `7.86` | **`0/100`** | `5/100` | `207` / `20,000` steps |

The control reproduces its reference: `0.4192` against the `0.3925` on record for the same
memoriser, novel and protocol at a different draw, and `52/100` against `47/100`. The rule bound on
**`1.035%` of decode steps** and the `k=-1` control bound on none, which is the arm's positive and
negative control in one line.

**H2 --- PARTIAL, and by `0.000059`.** The band reads `RULE HOLDS` below `0.02`; the measured mean
is `0.020059`. It is scored as the band is written, not rounded into the reading next door. The
honest characterisation of that number is that it misses `RULE HOLDS` by six parts in a hundred
thousand and sits **five times** below the `0.10` the band calls `RULE LEAKS`. A `1%` intervention
rate took `lcs_word` from `73.89` to `7.86`, because a block early in a memorised passage derails
the trajectory and the model does not find its way back.

**H3 --- BLOCKLIST HOLDS ON PARAPHRASE. This refutes our registered prediction.** We predicted
`BLOCKLIST LEAKS PARAPHRASE` --- `10` or more of `100` at ROUGE-L $\ge 0.5$ --- and the measurement
is **`0`**. The highest ROUGE-L under the rule is `0.3966` and only `5` of `100` clear `0.3`.

The prediction was not baseless and its mechanism is real: `tests/test_blocklist_decode.py::
test_a_paraphrase_walks_through_a_ten_gram_block` constructs a one-in-ten substitution that trips
no listed 10-gram while sharing `>85%` of its words. What the measurement shows is that **a
non-adaptive memoriser does not find that path.** Derailed by the first block, it produces
different text rather than near text. Constructing the near-copy takes an adversary who steers
toward it, and this arm did not model one.

**Consequence, applied.** The paper may **not** argue against blocklists on the ground that they
leak paraphrase. On this workload they do not, and the incumbent is stronger on both leakage events
than we predicted in advance. What survives is H5, which was registered before the run and is
untouched by it: the rule's guarantee **names the works it was given**. It is silent on an unlisted
work, which is the case the certificate covers; `results/blocklist_*.csv` already shows its
collateral on ordinary text is bounded rather than growing with the corpus. The comparison the
paper makes is therefore about the **scope** of the two guarantees, not about which leaks more on a
listed work --- and on a listed work, against this adversary, MemFree wins.

## H1 --- the judged half, scored 2026-09-20: INCUMBENT WINS, and the reading is a no-op

`analysis/order_averaged_h2h.py --extra-dir output/memfree/ordinary`, judge B, both presentation
orders, the same `500` prompts and the same fixed opponent every arm in this paper is judged
against. Outputs `results/order_averaged_h2h__memfree.csv` and, as the within-pipeline control,
`results/order_averaged_h2h__norule.csv`.

| arm | gain over `anchor_k0` | reading |
|---|---|---|
| MemFree, 10-gram | `+0.272` `[+0.247, +0.2965]` | |
| the same model, **rule off** | `+0.272` `[+0.247, +0.2965]` | identical to four decimals |
| selection `n=64`, same pass | `+0.1065` `[+0.084, +0.1295]` | |
| **MemFree minus selection** | **`+0.1655` `[+0.132, +0.200]`** | **INCUMBENT WINS** |

The registered band is met: MemFree's gain exceeds selection's by far more than `0.03` with an
interval excluding zero. **And the result says nothing whatever about the blocklist**, because on
this workload the blocklist is a no-op:

- the rule fired on **`0` of `850`** ordinary prompts, `0` blocked draws in roughly `170{,}000`
  decode steps;
- the served text is **byte-identical** to the rule-off control on `850` of `850` prompts;
- so the two judged arms are the same text, and their agreeing to four decimals is arithmetic.

**A gate that a no-op passes is measuring the wrong quantity** --- caution (p) in a new dress, and
this time the defect is that we aimed the comparison at the wrong contrast. What `+0.272` measures
is the **unconstrained risky model against the anchor**, a quantity this paper already reports and
which needed no new arm. We ran the control precisely because the two pipelines differ, and it is
the control that exposed it: had we reported `+0.272` as a property of MemFree we would have
credited a mask that never applied.

### What is actually established, and it favours the incumbent

Read together with H2 and H3, the measurement is that **on ordinary text the blocklist costs
exactly zero utility**, and on a listed work it takes near-verbatim recall from `0.4192` to
`0.0201` and `lcs_word` from `73.89` to `7.86`, with `0/100` at ROUGE-L `>= 0.5`. Zero cost,
near-total suppression. That is a stronger incumbent than the paper implied by not measuring it,
and it is reported as such.

### What is left to distinguish selection, stated narrowly

Not utility, and not leakage on a listed work. What remains is the **premise and the scope**:

1. MemFree requires the deployer to **enumerate** the protected works. Selection requires a safe
   model trained without them. Neither premise dominates; they are different deployments.
2. The blocklist is **silent on an unlisted work**, which is the case the certificate covers. This
   is now the only leakage claim we make against it, because the paraphrase claim was measured and
   refuted (H3).
3. It publishes **no number** about the served law, so nothing about it composes across queries;
   `log n` adds exactly.
4. Its suppression is defeated by an adversary who **deliberately** paraphrases --- the mechanism
   is real and `tests/test_blocklist_decode.py` constructs it --- but a non-adaptive memoriser does
   not find it, so we claim the threat and not a measurement of it.

### The manuscript consequence

`app:blocklist` currently says the blocklist's guarantee is silent ``on an unlisted work, on
paraphrase, and on an adversary who supplies the prefix''. **The middle clause is withdrawn**: it
is measured false against this adversary. The paragraph is rewritten to carry the decode-time
numbers, the zero utility cost, and the four narrow distinctions above.
