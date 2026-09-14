# Pre-registration: does the gain keep growing at the strongest anchor?

Committed **before the arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

Two numbers in the paper are currently attached to different anchors and are being read together.
The scaling result --- judged gain rising monotonically in `\log n` to `+0.142 [+0.097, +0.187]` at
`n = 64` --- is measured at **TinyComma-1.8B** only. The breadth result --- `+0.111
[+0.072, +0.148]`, the largest gain in the paper --- is measured at **Comma-7B** and only at
`n = 8`, because the breadth arm's grid stopped there. Nothing on record says what the strongest
anchor does at the largest `n`.

That gap matters in both directions. If the curve keeps rising, the paper's best number is not
`+0.142` but whatever Comma-7B reaches at `n=64`, and the abstract understates the mechanism. If it
turns over, the paper has found the **overoptimisation** its own Limitations names as an untested
risk: as `n` grows the argmax is optimised harder against the scorer's own errors, which for reward
models is known to turn the gain over rather than saturate it \citep{gao2023scaling}. Appendix~J
currently says only that the third reading is not excluded by Proposition~4. This arm can exclude
it or find it.

## What is run

`n \in \{1, 2, 4, 8, 16, 32, 64\}`, nested by seed order so no candidate is scored twice, drawn
from `common-pile/comma-v0.1-2t` on the **same 500 ordinary prompts** (200 neutral, 150 creative,
150 factual) at the same temperature and `200`-token cap as every other arm, self-paired so the
factory's shared-vocabulary requirement holds. The identical pointwise reward
(`log p("Yes") - log p("No")` from Qwen2.5-7B-Instruct on the one fixed template), the identical
two scoring judges, the identical unconstrained Llama-3.1-8B-Instruct opponent, and the entry gate
of `results/onset_prediction_selection_breadth.md` unchanged.

Nothing else changes from the `n=8` arm already on record, which is nested inside this one: the
`n \le 8` rows must reproduce it to within judging noise, and that is a check on the pipeline
rather than a result.

## Bands, committed before the run

Read on judge B, the registered scorer; judge C reported beside it and never substituted for it.
Gains are paired against the arm's own `n=1` control.

**G1 -- does the gain keep growing?** Comma-7B's `n=64` gain against its `n=8` gain of `+0.111`.

| reading | band |
|---|---|
| GROWS | the `n=64` gain exceeds `+0.111` by more than `0.03` |
| SATURATES | within `0.03` of it |
| OVEROPTIMISES | below it by more than `0.03` |

**Under GROWS the paper's headline number changes** and the abstract must carry the Comma-7B figure
rather than `+0.142`. **Under OVEROPTIMISES the Limitations paragraph on overoptimisation stops
being a caveat and becomes a measurement**, and Section 6 must say the mechanism has an operating
range with an upper end we found. Both consequences are fixed here so neither can be renegotiated.

**G2 -- is the curve monotone in `\log n`?** Spearman of `u` against `\log n` over the seven arms,
per judge. MONOTONE if `>= +0.9` in both, as it is at TinyComma (`+0.991`).

**G3 -- the nested check.** The `n \le 8` rows against the breadth arm's, which used the same
generations. Agreement to within `0.03` on the `n=8` gain is expected; a larger discrepancy means
the judging pass is not reproducible at this sample size and **G1 is not readable**, whatever it
says.

**G4 -- leakage is not re-run and not claimed here.** Comma-7B's near-verbatim recall is `0.0000`
at `n = 1, 8, 64` on the protected passages under the adversarial selector, already on record in
`results/selection_extraction_comma7b.csv`. This arm decodes only ordinary prompts and adds
nothing to that.

## Excluded alternatives (named now so they cannot be adopted afterwards)

1. Reporting the largest `n` that happens to look best. `n=64` is the endpoint and G1 is read there.
2. Changing the judge, the selector, the reward template, the prompt set or the opponent.
3. Dropping the `n \le 8` rows if G3 disagrees, rather than declaring G1 unreadable.
4. Quoting Comma-7B's gain in the abstract without the anchor named, under any reading.
5. Treating a gain at `n=64` as free: the certificate is `\log 64 = 4.16` nats pathwise, the
   measured KL is `3.175`, and the compute is 64 draws and 64 scoring passes per prompt.
6. Quoting a judged separation without its sample size (cautions (d), (e)).

## Scoring

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0.0 --safe-model-path common-pile/comma-v0.1-2t \
    --risky-model-path common-pile/comma-v0.1-2t --trajectories-per-prompt 64 \
    --cap-neutral 200 --cap-creative 150 --cap-factual 150 --cap-val 0 --cap-test 0 \
    --cap-attack-train 0 --max-new-tokens 200 --output-dir output/phase5/sel_comma7b_64
.venv/bin/python analysis/selection_scaling.py --gen-dir output/phase5/sel_comma7b_64 \
  --max-n 64 --reward-cache results/selection_rewards64_comma7b.csv \
  --tag _comma7b64 --out results
```

---

## Scoring log (appended after the run; nothing above this line is edited)

---

## Scoring, 2026-09-13 --- G3 fails, so G1 is not readable

`analysis/selection_scaling.py --gen-dir output/phase5/sel_comma7b_64 --max-n 64 --reward-cache
results/selection_rewards64_comma7b.csv --tag _comma7b64 --out results` ->
`results/selection_scaling_comma7b64.csv`, `500` ordinary prompts, seven nested arms, two judges.

| n | log n | KL nats | gain, judge B | gain, judge C |
|---|---|---|---|---|
| 2 | 0.693 | 0.1931 | +0.041 [+0.010, +0.073] | +0.062 [+0.028, +0.098] |
| 4 | 1.386 | 0.6363 | +0.060 [+0.021, +0.099] | +0.110 [+0.065, +0.156] |
| 8 | 2.079 | 1.2044 | +0.072 [+0.031, +0.112] | +0.145 [+0.100, +0.193] |
| 16 | 2.773 | 1.8351 | +0.103 [+0.063, +0.146] | +0.167 [+0.119, +0.213] |
| 32 | 3.466 | 2.4970 | +0.127 [+0.085, +0.170] | +0.189 [+0.142, +0.236] |
| 64 | 4.159 | 3.1745 | +0.173 [+0.130, +0.218] | +0.230 [+0.184, +0.277] |

### G3 -- FAILS on the registered scorer, and the committed consequence is applied

| judge | breadth arm, n=8 | this arm, n=8 | \|delta\| | band (0.03) |
|---|---|---|---|---|
| B (registered) | `+0.111` | `+0.072` | **0.039** | **FAIL** |
| C | `+0.155` | `+0.145` | 0.010 | pass |

The band is read on judge B, the registered scorer, as the top of this file requires. It fails, and
the band's own words are *"a larger discrepancy means the judging pass is not reproducible at this
sample size and **G1 is not readable**, whatever it says."* **So G1 is not read.** Excluded
alternative 3 forbids the obvious escape --- dropping the `n \le 8` rows and reading `n=64` anyway
--- and excluded alternative 4 forbids putting the number in the abstract. **The abstract does not
change**, because the consequence that would have changed it was attached to GROWS firing, and
GROWS did not fire; it was not reached.

**The premise of G3 was wrong when I wrote it, and that does not rescue it.** The band says the two
arms "used the same generations". They did not: this arm generated `64` trajectories per prompt into
`output/phase5/sel_comma7b_64`, a fresh draw, while the breadth arm used
`output/phase5/sel_comma7b_8`. The mean completion length gives it away --- `80.4` words here
against `99.5` there at the same `n=8`. So the two `n=8` numbers are independent draws judged in
independent passes, not the same candidates re-scored, and `0.039` is a **cross-pass** discrepancy
rather than a scoring bug. Rewriting the band around that after seeing the number is exactly what a
pre-registration is for preventing, so the band stands as written and G1 stays unread.

**It is corroboration, not an excuse.** Independently, on 2026-09-12, the second-pair head-to-head
judged two arms that ought to be identical --- the metered run's `k=0` arm and the selection run's
`n=1` draw, each one unconstrained completion from the same anchor on the same `500` prompts --- and
found `-0.034` `[-0.071, +0.004]` between them. That is the same quantity, measured a different way,
at the same size. **Two independent measurements now put this paper's cross-pass floor on a judged
gain at about `0.04`**, and every cross-arm gain comparison in the paper carries it whether or not
its within-pass interval shows it. Appendix A says so; it now has a second number.

### G2 -- MONOTONE

Spearman of `u` against `\log n` is `+1.000` over the seven arms in **both** judges, against a
committed `>= +0.9`.

### What this arm does support

Within one pass, on one set of generations, judged by one model, the gain rises monotonically from
`+0.041` at `n=2` to `+0.173` at `n=64` on the registered scorer and from `+0.062` to `+0.230` on
the second. That is a **within-arm** statement and the cross-pass floor does not touch it: the
mechanism is still climbing at `n=64` at the strongest anchor, for `3.17` nats of measured KL
against a `\log 64 = 4.16` pathwise certificate.

**OVEROPTIMISES is excluded on this evidence.** The curve does not turn over anywhere on the grid,
in either judge, so the reward-overoptimisation risk Limitations names is not observed up to `n=64`
at this anchor --- which is a bound on where it has been looked for, not a claim that it is absent.

### G4

Not re-run and not claimed, as committed. Comma-7B's near-verbatim recall stays `0.0000` at
`n = 1, 8, 64` from `results/selection_extraction_comma7b.csv`.
