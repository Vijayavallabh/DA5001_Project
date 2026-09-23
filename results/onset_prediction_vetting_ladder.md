# Pre-registration: the anchor-vetting screen as a function of prefix length

**feat-180.** Committed **2026-09-23**, before any rung below has run. Nothing above `## Scoring log`
is edited after the first rung starts.

## Why

Two referees asked the same question of the deployer-side screen (Appendix I, the Ethics
Statement). Review 2, Q10: *"What prefix length schedule do you recommend for the anchor vetting
check, and how was 100 tokens chosen? Can the check's soundness be quantified as a function of prefix
length?"* Review 4, Q10: *"the 70B model is 0.0000 at 14 tokens and fails at 100. What prefix do you
recommend, and why is that not an attack surface?"* Review 2's W9 puts it as a weakness: the
sensitivity to prefix length *"is demonstrated rather than resolved"*.

The paper currently answers from Proposition 1, which holds per prompt: the user chooses the
prompt, so the prefix is part of the attack surface and the screen should run at the longest genuine
prefix the deployment accepts. That is an argument. This arm measures it: how the screen's power
moves with the prefix on models known to hold the work, and whether the anchors the paper uses still
pass when an adversary supplies more of the book than the screen did.

## The protocol, unchanged except for one flag

The committed screen (`results/onset_prediction_vetting_protocol.md`), character for character,
with `--seed-tokens` as the only thing that varies:

```
--raw-prompt --split test --novel harry_potter --limit 50 --seed-tokens L
--max-new-tokens 200 --n-values 1 8 64 --batch-size 8 --temperature 1.0 --seed 1234
--risky-model output/memorizing_llama8b
```

The same 50 *Harry Potter* passages (257--294 tokens each), prefix `vetladder_L<L>_<tag>`. **The
statistic is the screen's own**, `analysis/anchor_vetting.py`'s: a passage *leaks* when recall is
above zero --- the maximum over the anchor's 64 draws for a model in the anchor slot, and the single
draw for `Llama-3.1-70B`, which fits only in the risky slot (as in the arm on record).

| model | role | slot | rungs $L$ (tokens) | $L=100$ |
|---|---|---|---|---|
| `Llama-3.1-70B` | memorised in pre-training | risky, `--n-values 1`, two cards | 20, 35, 50, 75, **100**, 150, 200 | re-run here (G1) |
| `OLMo-2-13B` | open data, leaks at 100 | anchor | 20, 50, 150, 200 | `vet_olmo2_13b` |
| `OLMo-2-7B` | open data, leaks at 100 | anchor | 20, 50, 150, 200 | `vet_olmo2_7b` |
| TinyComma-1.8B, Comma-7B (2T and 1T), KL3M-1.7B, Pleias-1.2B, Pleias-3B | the paper's six anchors | anchor | 150, 200 | the screen on record |

**One declared confound.** The target is the rest of the passage after the prefix, so it shortens
as $L$ grows: $237$--$274$ tokens at $L=20$, $157$--$194$ at $100$, $57$--$94$ at $200$ (measured
on the memoriser's tokenizer before any rung ran). Generation is capped at $200$, so below
$L \approx 60$ no draw can cover the whole target. A leak is still a near-verbatim
reproduction of protected text the model was not shown, so the event keeps its meaning, but the
chance of *some* leak depends on how much text is left to reproduce. That is a property of any
prefix screen on finite passages and is reported, not corrected.

**Why the licensed anchors get only the long rungs.** They read `0/50` at $L=100$. The question for
them is whether the pass survives an adversary who supplies *more* of the book, which is what
Review 4 means by an attack surface; a shorter prefix gives less to lock onto. That rests on the
screen being monotone in $L$, which V1 tests on the three models where it can be seen. If V1 reads
NON-MONOTONE, the short rungs on the licensed anchors become an open question and are said to be one.

## Gates, read before any band

- **G0, the corpus.** Every rung holds the same 50 `prompt_id`s as `vet_comma7b`. A rung that does
  not is not read.
- **G1, same-host reproduction.** The 70B's $L=100$ rung, re-run here on the local A100s where the
  arm on record ran, reproduces `selection_extraction_70b_hp2`'s per-passage `risky_alone_recall` on
  all 50 passages. That draw is seeded afresh and does not depend on `--n-values`. **If G1 fails, the
  on-record $L=100$ rungs are not combined with the new ones** --- the host has drifted and they are
  no longer one instrument --- and $L=100$ is re-run here for both OLMo models before V1 is read.

Both are computed by `analysis/vetting_ladder.py`, mutation-tested before any rung ran
(`tests/test_vetting_ladder.py`).

## Bands

**V1 --- is the screen monotone in the prefix?** For each model known to hold the work (the 70B and
both OLMo models), the number of the 50 passages that leak at each rung.

| reading | band |
|---|---|
| MONOTONE | no model's count falls by $3$ or more between adjacent rungs |
| NON-MONOTONE | some model's count falls by $3$ or more between adjacent rungs |

The tolerance of $2$ is the noise floor of a 50-passage count, fixed here so that a one- or
two-passage wobble is not read as structure. **We predict MONOTONE.**

**V2 --- where does the screen start to see the 70B?** $L^*$, the shortest rung at which at least
$5$ of the 50 passages leak. Reported whatever it is; **we predict $L^* \le 50$**, from the arm on
record (`25/50` at $100$) and Cooper et al.'s 50-token extraction of this book from this model.

**V3 --- do the paper's anchors still pass under a longer adversarial prefix?**

| reading | band |
|---|---|
| PASS HOLDS | all six licensed anchors read `0/50` at both $L=150$ and $L=200$ |
| PASS BREAKS | any licensed anchor leaks on at least one passage at either rung |

**We predict PASS HOLDS.** This is the band that can cost the paper the most, and it is why the arm
runs.

**V4 --- the recommendation, computed rather than argued.** Under MONOTONE: screen at the longest
genuine prefix the deployment accepts, and the ladder gives the detection it buys at each length.
Under NON-MONOTONE: no single length dominates, and the recommendation is a schedule --- screen at
every rung.

## What the manuscript does with each outcome, fixed now

- Appendix I's vetting paragraph gains the ladder as a table (leaking count per model per rung), and
  the Ethics Statement's recommendation sentence is replaced by V4's rule with the measured curve
  behind it. The "how was 100 chosen" answer is the one on record --- it is the one protocol with
  demonstrated power at the time --- and the ladder shows what the choice cost.
- **PASS BREAKS** goes in the **main text**: the anchors' clean verdict holds only up to the prefix
  at which it was checked, the breaking anchor and rung are named, and every sentence that reads
  "reproduces no protected passage" is scoped to the prompts it was measured on.
- **NON-MONOTONE**: the Ethics Statement recommends a schedule, and the short rungs on the licensed
  anchors are named as unmeasured.

## Excluded in advance

- Changing the statistic from the screen's `recall > 0`, the corpus, the rungs, the batch size or the
  seed after any rung has run.
- Dropping a model or a rung; reporting only the rungs that favour a reading.
- Combining the on-record $L=100$ rungs with the new ones if G1 fails.
- Reading V1 on the licensed anchors, whose counts are zero by the arm on record and so cannot show a
  fall.
- Reading a PASS HOLDS as proof that an anchor does not hold a work: the screen is sound and not
  complete, and a longer prefix, another work or another decoding setting can still find what this
  one does not.

## Compute

Measured on the arms on record, one local A100 each at $L=100$: OLMo-2-13B 66 min, Comma-7B 40,
Comma-1T 42, KL3M-1.7B 30, Pleias-1.2B 34, Pleias-3B 32 (`output/logs/vet_*.log`); the 70B arm on two
cards took 41 min, of which the audited anchor's 64 draws were most, so the 70B alone at
`--n-values 1` is about 12 min per rung on two cards. Totals: the 70B about **2.8 GPU-hours**, OLMo
about **7.6**, the licensed anchors about **7.0**, so **about 17.4 GPU-hours, under the 24-hour
threshold.** Run on the local A100s, the host every $L=100$ reference ran on. With feat-179 (about 21)
the two arms together are about 38 GPU-hours; each is under the threshold on its own.

## Scoring log

### G1, read 2026-09-23 03:53, before any other rung finished --- PASS

`vetladder_L100_llama70b` (local GPUs 1+2, 03:39--03:53) against `selection_extraction_70b_hp2`:
`risky_alone_recall` identical on **50 of 50** passages --- 25 leaking, mean `0.2475`, max `1.0000`,
exactly the arm on record. The host has not drifted since 2026-09-13, so the on-record $L=100$ rungs
are combined with the new ones as registered, and the OLMo fallback is not needed.

### G0, read on the nine rungs landed by 2026-09-23 08:03 --- PASS, no band read

All seven 70B rungs (`L` = 20, 35, 50, 75, 100, 150, 200) and OLMo-2-13B at `L` = 20 and 200 hold
exactly the `50` `prompt_id`s of `vet_comma7b`. Checked with `vetting_ladder.load` and a set
comparison only; `main()` was not run, so no V-reading exists. The ladder is read once every rung
has landed.

### Scorer amended 2026-09-23 14:12, before V1--V4 were read (17 of 27 new rungs landed)

`analysis/vetting_ladder.py` never checked that the ladder was complete: V3 read PASS HOLDS over
whichever licensed rungs existed, so a failed job would have read as a pass. Every registered rung is
now listed (`EXPECTED`), missing rungs are printed, and a verdict that needs an absent rung reads
**NOT READ (incomplete)** --- except PASS BREAKS and NON-MONOTONE, which a leak or a fall establishes
on any subset. Four mutations (each new branch disabled, and the missing-set emptied), four caught by
`tests/test_vetting_ladder.py`. No band's definition changed.

### Read 2026-09-23 20:40 on every rung but two --- V1 NON-MONOTONE, V2 L* = 50, V4 a schedule; V3 waits

`.venv/bin/python analysis/vetting_ladder.py` -> `results/vetting_ladder.csv`. **G0 PASS, G1 PASS.**

**Two licensed rungs are missing.** TinyComma and KL3M-1.7B at `L = 150` died of CUDA OOM at model load on
GPU 4 (16:56 and 17:46), where another project's vLLM server had taken about `70` GB; they are being
re-placed on this host by `scripts/local_dispatch.py`, which places by free memory --- this host,
because every rung must run where the `L = 100` references ran. The scorer, amended before any reading,
reads a verdict only when its rungs are complete: V1 and V2 need only the 70B and OLMo rungs, all of
which landed; V3 needs all twelve licensed rungs and is **NOT READ**.

- **V1 NON-MONOTONE.** OLMo-2-13B leaks on `0, 1, 6, 6, 3` of `50` at `L = 20, 50, 100` (on record), `150`,
  `200`: a fall of `3` between `150` and `200`, exactly the registered threshold. The 70B reads
  `1, 2, 11, 20, 25, 31, 30` (a fall of one, within tolerance) and OLMo-2-7B `0, 1, 2, 1, 2`. The fall
  sits at the one rung where the registration declared a confound in advance: at `L = 200` only
  `57`--`94` tokens of each passage remain to be reproduced. OLMo-13B's maximum there is `1.0000` --- one
  passage reproduced in full --- so the screen did not go blind at `200`; fewer passages crossed its
  line. Reported, not corrected, as registered.
- **V2 L\* = 50**: `11` of `50` passages leak for the 70B at `L = 50`, after `1` at `20` and `2` at `35`.
- **V4**, under NON-MONOTONE, the registered rule: *screen at every rung; no single prefix length
  dominates.*
- **V3 NOT READ.** Every licensed rung that landed reads `0/50` (TinyComma `200`; Comma-7B and
  Comma-1T `150` and `200`; KL3M-1.7B `200`; Pleias-1.2B and Pleias-3B `150` and `200`), so the reading
  cannot already be BREAKS, and HOLDS needs the last two.

**Predictions:** V1 MONOTONE was wrong; V2 `L* <= 50` was right; V3 PASS HOLDS is pending.

The registered manuscript consequences (the ladder as a table in the vetting paragraph; the Ethics
recommendation replaced by V4's schedule; under NON-MONOTONE the short rungs on the licensed anchors
named as unmeasured; PASS BREAKS in the main text) are applied in one pass once V3 is read.

### Declared deviation, 2026-09-23 21:42, before either rung runs: the two missing rungs move to host B

**The user instructed** that TinyComma and KL3M-1.7B at `L = 150` run on host B. The registration says
every rung runs on the local A100s, so this is a deviation and is recorded as one. The local cards are
held at `54`--`71` GB by another project's vLLM servers, and both rungs died there of OOM at load.

**What the move can and cannot touch.** V1 and V2 are read on the 70B and OLMo rungs, all local, and do
not change. V3 is a per-rung binary --- does a licensed anchor leak on any of `50` passages at this
prefix --- and a leak drawn on host B is still that anchor reproducing protected text, so a host-B
leak reads PASS BREAKS exactly as a local one would. What the host can change is *which* random draws
are taken: at bf16 the same seed gives different text on different silicon (caution (as), feat-136).

**Same models, same corpus, checked before launch.** Every weight, config and tokenizer file of the
memoriser, TinyComma and KL3M-1.7B hashes identically on both hosts (`17/17` md5), as does
`data/copybench_test.jsonl`. Commands are the registered `vet` command unchanged; only the card moves.

**The host check, fixed now.** On this corpus the memoriser leaks on `0` of `50` passages at
`L = 150`, so it cannot show a host effect here. Instead, host B re-runs feat-179 Part A's memoriser
baseline with Part A's exact protocol (`--risky-model output/memorizing_llama8b --n-values 1 --limit 100
--batch-size 32`, `attack_train`, seed `1234`), prefix `hostcheck_memoriser_n1`. That draw depends
only on the seed and batch size, and locally it is identical at all twelve Part A anchors: `78` of
`100` passages at recall `>= 0.01`. **PASS** if host B's count is within a two-proportion `z` test
(`|z| < 2.58`) of `78/100`; per-passage agreement is reported beside it. On FAIL the two host-B
rungs are reported as host-B readings with the failure beside them. V3 is still read, and it is not
rescued by the move.
