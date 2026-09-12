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
