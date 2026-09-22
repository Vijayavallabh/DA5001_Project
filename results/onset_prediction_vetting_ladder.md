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
