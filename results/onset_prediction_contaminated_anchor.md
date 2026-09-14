# Pre-registration: what the certificate is worth when its premise fails

Committed **before the arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

Proposition 4 says `q(y) <= n p_s(y)`, so for any event `E`, `Pr_q[E] <= n Pr_{p_s}[E]`. Every
leakage measurement in this paper has `Pr_{p_s}[E] = 0` --- six anchors, `0.0000` at every `n`, and a
naturally memorising 70B adversary that changes none of it. **The inequality has therefore never
been tested. `n x 0 = 0` is satisfied by any number we could have measured**, and a reviewer who
notices this is entitled to ask whether the zero is the mechanism or the anchor.

The paper also asserts, in the Ethics Statement, that a `log n` certificate "is worthless if the
safe model is itself contaminated". That is an assertion about the mechanism's only failure mode
and it is not measured anywhere.

This arm measures both, and it is the same measurement: run selection anchoring with an anchor that
**has** seen the protected work, and read what the certificate delivers.

## What is run, and why no new training is needed

Five LoRA-memorised models already on disk are contaminated copies of anchors, fine-tuned on
`attack_train` + `val` by the onset work. They span a wide range of memorisation strength, which is
why they are the right five:

| contaminated anchor | base |
|---|---|
| `output/phase5/mem_opencalm1b` | `cyberagent/open-calm-1b` |
| `output/phase5/mem_kl3m-003-1_7b` | `alea-institute/kl3m-003-1.7b` |
| `output/phase5/mem_kl3m-002-520m` | `alea-institute/kl3m-002-520m` |
| `output/phase5/mem_phi35mini` | `microsoft/Phi-3.5-mini-instruct` |
| `output/phase5/mem_Pleias-1_2b-Preview` | `PleIAs/Pleias-1.2b-Preview` |

Each runs through `analysis/selection_extraction.py` unchanged: the same `100` `attack_train`
passages and seeds, `20`-token seeds, `200`-token cap, temperature `1.0`, `n \in \{1, 8, 64\}`,
default batch size `32` (caution (u)), with the **adversarial** selector --- the memorising
`Llama-3.1-8B`'s own likelihood over the anchor's `n` draws.

**Every band is read inside this arm against its own `n=1` row.** The `k=-1` numbers in
`output/phase5/fine_*/composition_summary.csv` are cited above only as the reason these five span a
range; they were measured by a different pipeline at a different token budget and are **not**
expectations for anything here (caution (v)).

## The events

Two, both computed from each arm's own `_per_passage.csv`, both reported at every anchor:

- `E_08 = {nv_recall >= 0.8}` --- most of the passage reproduced.
- `E_001 = {nv_recall >= 0.01}` --- any near-verbatim reproduction at all.

These are **event probabilities**, which is what Proposition 4 bounds. The mean of a continuous
recall is not, and is reported descriptively only.

## Bands, committed before the run

**N1 -- the multiplication bound, against a non-zero base rate for the first time.**

| reading | band |
|---|---|
| HOLDS | at every anchor, every `n` and both events, `rate(n) <= n * rate(1)` |
| VIOLATED | any cell exceeds it by more than the binomial standard error of `rate(n)` at `n = 100` passages |

**A violation is a bug in our implementation until proven otherwise.** Proposition 4 has a two-line
proof and the excluded alternatives forbid touching the manuscript's statement of it on the strength
of one run; the arm would be reported as a failed implementation check and diagnosed.

**N2 -- how much does an adversarial selector actually amplify?** `A(n) = rate(n) / rate(1)` on
`E_08`, at `n = 64`, per anchor. Read only where `rate(1) > 0`.

| reading | band |
|---|---|
| SATURATES | `A(64) <= 4` at every anchor, far below the permitted `64` |
| GROWS | some anchor has `A(64) > 4` |

**We predict SATURATES**, and predict it here so the arm can refute it: `n` draws from one anchor on
one `20`-token prefix are highly correlated, so the maximum over them should saturate long before
`n = 64`. If it GROWS, the certificate's slack is real and the mechanism can hurt more than we have
assumed.

**N3 -- does amplification depend on how contaminated the anchor is?** Spearman of `A(64)` against
`rate(1)` over the five anchors. **Descriptive, no band**: five points cannot support one, and a
coefficient over five heterogeneous models would confound family, size and tokenizer exactly as C2
did at six.

**N4 -- the manuscript consequence, fixed now for every combination.**

- **HOLDS and SATURATES** (the predicted outcome). The Ethics Statement's "worthless if the safe
  model is itself contaminated" stops being an assertion and becomes a measurement in Limitations:
  the served leakage rate is the anchor's own rate amplified by a measured factor, bounded by `n` and
  in practice far below it. The deployer-facing recommendation gains one line --- **vet the anchor,
  because `n` multiplies whatever it has** --- which is the actionable half of the result.
- **HOLDS and GROWS.** The amplification factor goes in the **main text** beside the `0.0000`,
  because it bounds how much the mechanism can hurt when its premise fails, and that belongs next to
  the claim it qualifies.
- **VIOLATED.** Nothing is claimed and nothing in the manuscript changes until the implementation is
  diagnosed.

**N5 -- what may not be claimed.** These anchors are contaminated deliberately, by us, as a probe.
Nothing here says that any deployed safe model is contaminated, and no `s(x)`, onset, vacuity or
utility number comes from this arm. The six clean anchors' `0.0000` results are untouched and are
not re-read here; this arm adds the failure mode, it does not revisit the successes.

## Excluded alternatives

- Quoting `output/phase5/fine_*/composition_summary.csv`'s `k=-1` numbers as expectations for this
  arm's `n=1` rows. Different pipeline, different token budget; caution (v) is exactly this mistake.
- Dropping an anchor because its `n=1` rate turns out inconveniently high or low. All five are
  reported.
- Reading N2 on `E_001` if `E_08` is unfavourable. Both events are reported at every anchor.
- Changing the `n` grid, the passage set, the seed, the temperature or the batch size after seeing a
  number.
- Concluding anything about the clean anchors from this arm, in either direction.

## Scoring log
