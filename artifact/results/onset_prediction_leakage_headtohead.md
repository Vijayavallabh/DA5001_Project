# Pre-registration: the two mechanisms' leakage, on one set of passages with one control

Committed **before either arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

The paper measures both mechanisms' leakage and never on one footing. Section 4's onset says where
extraction becomes measurable for the **metered** decoder as `k` rises; Section 6 says the
**selection** decoder reproduces nothing at any `n <= 64`. They are two separate measurements with
two separate controls, and a reviewer is entitled to ask whether the comparison survives being made
in one place.

It does not survive it automatically. The two pipelines disagree about their own shared baseline:
the memorising model alone reads `0.4921` in `results/composition_8b_kl.csv` and `0.3925` in
`results/selection_extraction.csv`, on the same split, the same `--limit 100`, the same
`--seed-tokens 20` and the same temperature. **That gap is the reason for this arm.** The most
likely cause is generation length --- `composition_attack.py --modes single` decodes
`max_new_tokens = |target|`, about `260` tokens, and `selection_extraction.py` decodes `200` against
the same full-length reference, so its `nv_recall` cannot exceed about `0.77` by construction.

## What is run

One anchor (the audited `jacquelinehe/tinycomma-1.8b-llama3-tokenizer`), one risky model
(`output/memorizing_llama8b`), the **same 100 `attack_train` passages and seeds** both arms already
use, temperature `1.0`, seed `1234`:

- **Arm A, metered.** `analysis/composition_attack.py --modes single --k-values -1 0 0.5 1 3 20`,
  KL accounting, no bank cap, prefix debt on --- the deployed configuration. Reports
  `nv_recall_mean`, `nv_recall_max`, `lcs_word_mean` and the **realised** spend in nats.
- **Arm B, selection.** `analysis/selection_extraction.py --n-values 1 8 64 --max-new-tokens 300`,
  the adversarial selector (the memorising model's own likelihood), at the generation length Arm A
  gets rather than the `200` on record.

Arm A writes `results/h2h/`, Arm B writes `results/leakage_len300*.csv`. Neither existing CSV is overwritten.

## G0 --- the comparability gate, read before any band

Same model, same passages, same seeds, same length: the two `k=-1` rows must now agree.

| reading | band |
|---|---|
| **MATCHED** | `\|nv_recall_mean(A, k=-1) - nv_recall_mean(B, n=-1)\| <= 0.02` |
| **UNMATCHED** | anything larger |

**Under UNMATCHED no joint table is built.** The paper keeps the two measurements separate, and the
residual is reported in Appendix~J as a limit on how far the two can be compared. This gate can
cost us the headline and is written that way on purpose: length was a hypothesis, not a diagnosis,
and if it is wrong something else differs that we have not identified.

## Bands, committed before the run

**M1 --- where does the metered decoder start leaking?** The smallest `k` in the grid whose
`nv_recall_mean` exceeds `0.005`, and the realised spend in nats at that `k`.

| reading | band |
|---|---|
| LEAKS IN GRID | some `k <= 20` exceeds `0.005` |
| CLEAN IN GRID | none does |

**M2 --- does selection still reproduce nothing at the longer length?** Arm B at `n=64`.

| reading | band |
|---|---|
| STILL ZERO | `nv_recall_mean = nv_recall_max = 0.0000` |
| NONZERO | anything else |

**M3 --- the prediction, committed so the arm can refute it.** We expect **MATCHED**, **LEAKS IN
GRID** and **STILL ZERO**, and concretely that the metered decoder's first leaking `k` costs more
than `100` realised nats against selection's `\log 64 = 4.159`. If the metered decoder is CLEAN IN
GRID the comparison is a draw on leakage and the paper says so --- the constructive claim then
rests on the spend and the utility, which is where it already rests.

**M4 --- the reading the paper must carry**, fixed now in all cases:

- **MATCHED, LEAKS, STILL ZERO.** Section 6's leakage paragraph gains one sentence putting both
  mechanisms on the same 100 passages with one control, and the ratio of realised spends.
- **MATCHED, CLEAN, STILL ZERO.** The sentence says both are clean on this grid and the separation
  is in what they spend, not in what they leak. No leakage claim is upgraded.
- **NONZERO.** Section 6's `0.0000` acquires an explicit generation-length qualifier, in the main
  text and not a footnote, and the abstract's "reproduces none of the protected passages at any
  `n <= 64`" is re-scoped to the length at which it was measured.
- **UNMATCHED.** No joint table. Appendix~J reports the residual.

## Excluded alternatives

- Changing the `k` grid, the `n` grid, the passage set, the seed or the temperature after seeing a
  number.
- Building the joint table under UNMATCHED, or quoting a ratio across the two existing CSVs whose
  controls disagree.
- Substituting `lcs_word` for `nv_recall` in G0 after the fact. `lcs_word` is reported for both
  arms and is descriptive here; the gate is on `nv_recall_mean`, fixed now.
- Reporting Arm A's oracle or chained modes. They are composition attacks with many queries and
  have no counterpart in Arm B; only `single` is comparable.

## Scoring log

## Scoring, 2026-09-14 --- G0 reads UNMATCHED, and no joint table is built

Run: `scripts/run_leakage_headtohead.sh 4`, GPU 4, `[l2l] metered exit=0`, `[l2l] selection exit=0
at 10:02`. Outputs `results/h2h/composition{,_summary}.csv` and `results/leakage_len300{,_per_passage}.csv`.

### G0 --- the comparability gate

| arm | `k=-1` / `n=-1`, the memoriser alone | |
|---|---|---|
| A, `composition_attack.py --modes single` | `0.4921` | `max_new_tokens = |target|`, mean about `260` |
| B, `selection_extraction.py --max-new-tokens 300` | `0.4428` | fixed `300` |

`|0.4921 - 0.4428| = 0.0493` against a registered `0.02`. **UNMATCHED.**

**The pre-registered consequence applies: no joint table is built.** The paper keeps the two
leakage measurements separate, each against its own control, and this residual goes in
Appendix~J as a limit on how far they can be compared. No ratio is quoted across the two CSVs.

### The hypothesis was half right, which is why the gate was a gate

Generation length was the named suspect and it accounts for **half** the discrepancy: lengthening
Arm B from `200` to `300` tokens moved its control from `0.3925` to `0.4428`, closing `0.0503` of
the original `0.0996` gap in the predicted direction. That is a real effect --- at `200` tokens
against a full-length reference, `nv_recall` is capped near `0.77` by construction --- and it is
not the whole effect.

Two candidates remain for the residual `0.0493`, and this arm cannot separate them:

1. **The draw changed.** `generate()` consumes the RNG per sampling step, so raising
   `max_new_tokens` from `200` to `300` gives a different sampled trajectory even at a fixed seed
   and batch size --- the same mechanism as caution (u), on a different knob. Arm B's `k=-1` is a
   sampled arm, so part of the `0.0503` move is a fresh draw rather than extra tokens.
2. **The lengths are still not matched.** Arm A decodes each passage's own `|target|` (mean about
   `260`, max `296`); Arm B decodes a flat `300` for every passage. They agree on no passage
   exactly.

Distinguishing them needs a third run at a per-passage length, which this pre-registration's
excluded alternatives forbid after seeing a number, and which would in any case be a third draw.

### What is not claimed

No joint table, no cross-arm ratio, and no statement of the form "the metered decoder leaks $X$
where selection leaks $0.0000$" built on these two CSVs. Arm A's own numbers stay internally
valid --- at `k=20` it recovers `0.4759` mean recall for `761.3` realised nats against the
unconstrained model's `0.4921`, and at `k \in \{0.5, 1\}` it recovers `0.0000` --- and Arm B's stay
internally valid, and the paper reports each beside its own control as it already did.

### The one thing the arm settles

It settles that the `0.4921`-against-`0.3925` discrepancy noted in the pre-registration is **not**
a defect in either pipeline: both measure a sampled memoriser correctly, and they differ because
they decode different numbers of tokens and therefore draw different trajectories. That was worth
knowing before a reviewer found the two numbers in the released CSVs and drew a worse conclusion.
