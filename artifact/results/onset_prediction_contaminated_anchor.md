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

## Scoring, 2026-09-14 — read at twelve anchors, superseding the five-anchor reading

`analysis/contaminated_anchor.py --out results` -> `results/contaminated_anchor.csv`. The twelve
arms are `scripts/run_contaminated_anchor.sh`; the two Mixtral anchors carry the declared deviation
recorded in `results/onset_prediction_contaminated_anchor_twelve.md` (batch `32` with
`--experts-impl eager`, committed before either produced a number).

**Every anchor reproduces the memoriser control at `0.3925 / 0.8154 / 78.0%`**, bit for bit,
including both MoE arms --- twelve independent confirmations that the selector and the passage set
are the same across arms.

### The informative event, `E_08 = {nv_recall >= 0.8}`

| anchor | rate(1) | rate(8) | rate(64) | `A(64)` | bound at `n=8` |
|---|---|---|---|---|---|
| kl3m-002-170m, kl3m-003-1.7b, kl3m-002-520m, kl3m-003-3.7b, open-calm-1b, open-calm-3b | `0.000` | `0.000` | `0.000` | --- | `0.000` |
| phi35mini | `0.000` | `0.010` | `0.010` | --- | `0.000` |
| Llama-3.2-1B | `0.020` | `0.040` | `0.080` | `4.00` | `0.160` |
| Llama-3.2-3B | `0.030` | `0.090` | `0.110` | `3.67` | `0.240` |
| Pleias-350M | `0.030` | `0.050` | `0.040` | `1.33` | `0.240` |
| Pleias-1.2B | `0.050` | `0.050` | `0.050` | `1.00` | `0.400` |
| Qwen2.5-7B | `0.100` | `0.130` | `0.150` | `1.50` | `0.800` |

### N1 --- VIOLATED as registered, and the violation is the band, not the theorem

One anchor trips it: phi35mini reads `0.010` at `n=8` and `n=64` against a bound of `n * 0.000 =
0.000`, exceeding it by `5e-5` --- the binomial standard error at `rate = 0.01, N = 100` is
`0.00995`. **It is one passage.** `bookmia.07.16` crosses `0.8` in the served string at `n=8`, and
its own `n=1` draw sits at `0.1581`.

The registered rule is that a violation is an implementation bug until proven otherwise. It is
proven otherwise, and the diagnosis is not flattering to the band: **where `rate(1)` is exactly
`0`, the empirical bound collapses to `0` and any non-zero rate at any `n` registers as a
violation.** Proposition 4 bounds true probabilities; a true `p_s` of a few thousandths is entirely
consistent with `0/100` at `n=1` and `1/100` at `n=8`. The test has no power there, and **seven of
twelve anchors are there.**

**We do not re-specify the band to rescue the reading.** N1 is reported as VIOLATED at the letter
of what was committed, with the cause. What the arm does establish is the half that has power: at
the **five** anchors where `rate(1) > 0`, the bound holds with a wide margin at every `n` and both
events --- `rate(8)` of `0.04`-`0.13` against bounds of `0.16`-`0.80`. That is the first
non-vacuous test of `Pr_q[E] <= n Pr_{p_s}[E]` in this paper, and it passes.

### N2 --- SATURATES, at the boundary, and the boundary is thin

The largest `A(64)` on `E_08` is **exactly `4.00`**, at Llama-3.2-1B, against a registered band of
`<= 4`. It reads SATURATES by equality. The honesty owed here is the count: that ratio is `8`
passages over `2`. **One more passage at `n=64` would have made it `4.5` and flipped the reading to
GROWS.** Quote the band with its resolution or not at all.

What survives the fragility is the magnitude that matters: across the five readable anchors
`A(64)` runs `1.00`-`4.00` against a **permitted** `64`. An adversarial selector maximising the
memorising model's own likelihood over `64` draws realises at most about `6%` of the amplification
the certificate allows.

### N3 --- AMPLIFIES LESS, and this is what the twelve-anchor extension bought

Spearman of `A(64)` against `rate(1)` over the anchors where it is defined: **`-0.700`**, inside
the `<= -0.6` band committed in
`results/onset_prediction_contaminated_anchor_twelve.md`. **The prediction holds**: selection adds
most where the anchor leaks least. Pleias-1.2B at `rate(1) = 0.050` amplifies by `1.00`;
Llama-3.2-1B at `0.020` amplifies by `4.00`.

It is read over **five** anchors, not twelve, because `A` is undefined wherever `rate(1) = 0`. That
is far short of what the extension was written to deliver and is stated plainly --- **but the
extension is what made it readable at all**: of feat-111's original five anchors, exactly **one**
(Pleias-1.2B) has a non-zero `E_08` base rate. One point is not a correlation. Five is thin and is
labelled thin.

### The other event, `E_001 = {nv_recall >= 0.01}`, is reported and carries nothing

Base rates run `0.57`-`0.98`, so `n * rate(1)` exceeds `1` at every `n >= 2` and the bound is
vacuous by construction. Prop 4 holds at every cell trivially. `A(64)` runs `1.01`-`1.74`.
Reported because the pre-registration requires both events at every anchor; read for nothing.

### N4 --- the manuscript consequence, and what it is NOT

N4's HOLDS-and-SATURATES branch does not fire cleanly, because N1 does not read HOLDS. What is
applied is narrower than that branch would have licensed, and the difference is deliberate:

- The Ethics Statement's *"worthless if the safe model is itself contaminated"* becomes a
  **measurement** in Limitations. That sentence never depended on N1: it is about what the served
  rate is, not about whether the bound is tight.
- The paper states the amplification range `1.0`-`4.0` against a permitted `64`, **with the number
  of anchors and the counts behind it**, and the deployer line: `n` multiplies whatever the anchor
  already has, so the anchor is what must be vetted.
- The paper does **not** claim Proposition 4's bound is confirmed. It says where the test has power
  it holds, where it does not the base rate is zero, and that seven of twelve anchors are in the
  second case.

### N5 --- unchanged

These anchors are contaminated deliberately, by us, as a probe. Nothing here says a deployed safe
model is contaminated, no `s(x)`, onset, vacuity or utility number comes from this arm, and the six
clean anchors' `0.0000` results are untouched and not re-read. Four of the twelve
(`llama32-1b`, `llama32-3b`, `phi35mini`, `qwen25-7b`) are instruction-tuned or
undisclosed-corpus models and are contamination levels only.
