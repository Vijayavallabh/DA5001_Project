# Pre-registration: is Proposition 3's measured content one pair's, or the mechanism's?

Committed **before either arm is generated**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

Proposition 3 says a bucket-metered decoder's realised spend is the cost of **imitating** the risky
model on the steps where the bucket is slack, `Theta(T)` nats independent of the utility bought.
The proposition is proved. Its *measured* content --- `results/imitation_cost.csv` --- is one
(anchor, risky) pair: TinyComma-1.8B against Llama-3.1-8B-Instruct. The constructive half of the
paper stopped being a single setup today (`results/onset_prediction_selection_breadth.md`, B1
GENERALISES over four anchors); the vacuous half has not, and a reviewer who accepts the argument
for one is entitled to ask it of the other.

What is at stake is narrower than for the constructive claim, and the file says so before the
numbers: the proposition is a theorem, so no measurement can refute it. What a second and third
pair can show is whether the *shape* the audited pair exhibits --- an imitation rate that saturates
in `k`, a realised spend that converges to it, a cumulative spend linear in the step index, and a
decoder that ends up serving `p_r` unchanged at almost every step --- is a property of the
mechanism or of that pair.

## The pairs

Anchored decoding fuses two distributions over one shared vocabulary, so a second pair needs a
shared tokenizer. Llama-3.2 uses Llama-3.1's `128{,}256`-token vocabulary, which gives two pairs
against the same risky model the audited pair uses:

| pair | anchor | risky | why |
|---|---|---|---|
| A | `meta-llama/Llama-3.2-1B` (base) | `meta-llama/Llama-3.1-8B-Instruct` | the audited pair's shape: a small base anchor against an instruction-tuned model eight times its size |
| B | `meta-llama/Llama-3.2-3B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` | same family, both instruction-tuned, so the two distributions are close |

**Neither anchor is a legitimate \emph{safe} model for the copyright setting** --- both are trained
on undisclosed data and may have read the protected works --- and nothing about near-access-freeness
is claimed for them. They are here only to vary the pair's divergence while holding the mechanism,
the workload and the budget grid fixed, which is what Proposition 3's measured content is about.
This is stated now so it cannot be quietly upgraded later.

`k \in \{0.5, 1, 3, 20\}` on the same `500` ordinary prompts (200 neutral, 150 creative, 150
factual) at the same temperature and `200`-token cap as every other arm, with the mandatory `k=-1`
and `k=0` baselines on the same prompts and seeds. Zero-GPU analysis afterwards, from the per-step
logs.

## Bands, committed before the run

Write `r_imit(k)` for the imitation rate in nats per token (the mean charge on the steps the log
marks risky-unchanged) and `r_real(k)` for the realised spend rate. The audited pair's values are
on record: `r_imit` runs `0.0337` at `k=0.1` to `0.8482` at `k=5`, `r_real` tracks it to within
`0.4\%` from `k=1` upward, the median within-trajectory `R^2` of cumulative spend on step index is
`0.99` at every budget, and at `k=20` the decoder serves `p_r` unchanged at `99.95\%` of steps.

**I1 -- does the imitation rate saturate?** Per pair, `r_imit(20) / r_imit(3)`.

| reading | band |
|---|---|
| SATURATES | within `1.10` |
| STILL CLIMBING | above `1.10` |

**I2 -- is the realised spend the imitation cost?** Per pair, `|r_real(k)/r_imit(k) - 1|` at
`k \in \{3, 20\}`.

| reading | band |
|---|---|
| SPEND IS IMITATION | below `0.05` at both budgets |
| PARTLY | below `0.05` at one |
| NOT THE IMITATION COST | above `0.05` at both, and Proposition 3's measured content is the audited pair's |

**I3 -- is the spend linear in the step index?** Median within-trajectory `R^2` at every `k`.
LINEAR if `>= 0.95` at every budget in both pairs, otherwise reported per budget.

**I4 -- does the decoder end up as the risky model?** Fraction of steps serving `p_r` unchanged at
`k=20`. AS PREDICTED if `>= 0.99`.

**I5 -- the contrast, and the one thing here that could surprise.** Pair B's anchor and risky model
are the same family and both instruction-tuned, so `r_imit` should be far below pair A's. Predicted
before the run: **`r_imit(3)` for B is below half A's.** If instead the two pairs have similar
imitation rates, the rate is not measuring the distance between the models and the reading of
`results/imitation_cost.csv` in the paper is wrong. No band on the ratio itself beyond the factor of
two, because nothing on record calibrates it.

## Excluded alternatives (named now so they cannot be adopted afterwards)

1. Dropping a pair, a budget, or a prompt class after seeing any number.
2. Describing either anchor as a safe model, or quoting any leakage or certificate number from
   these arms. They carry none, by construction, and I3/I4 are about the decoder's spend only.
3. Changing the budget grid, the workload or the cap between pairs or against the audited arm.
4. Reporting the rates without the mandatory `k=-1` and `k=0` baselines.
5. Reading a confirmation of Proposition 3 into any of this. The proposition is proved; only the
   generality of its measured shape is at issue.
6. Using an aggregate counter where the per-step log is the source of truth (AGENTS caution (l)).

## Scoring

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<free card> HF_HUB_OFFLINE=1 \
  HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python h1.py --k-values -1 0 0.5 1 3 20 \
    --safe-model-path <anchor> --risky-model-path meta-llama/Llama-3.1-8B-Instruct \
    --trajectories-per-prompt 1 --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
    --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 \
    --output-dir output/phase5/imit_<tag>
.venv/bin/python analysis/imitation_cost.py --dirs output/phase5/imit_<tag> --tag _<tag> --out results
```

---

## Scoring log (appended after the run; nothing above this line is edited)

---

## Scoring, 2026-09-12 (appended; nothing above is edited)

```
# per pair: k in {-1, 0, 0.5, 1, 3, 20} on the same 500 ordinary prompts, one trajectory each
.venv/bin/python h1.py --k-values -1 0 0.5 1 3 20 --safe-model-path <anchor> \
  --risky-model-path meta-llama/Llama-3.1-8B-Instruct --trajectories-per-prompt 1 \
  --cap-neutral 200 --cap-creative 150 --cap-factual 150 --cap-val 0 --cap-test 0 \
  --cap-attack-train 0 --max-new-tokens 200 --output-dir output/phase5/imit_<tag>
.venv/bin/python analysis/imitation_cost.py --dirs output/phase5/imit_<tag> \
  --tag _<tag> --min-trajectories 20 --out results
```

`analysis/imitation_cost.py` gained `--dirs`/`--tag` for this; the audited arm reproduces
**byte-identically** with the new flags (`diff` against the committed CSV is empty), so nothing on
record moved.

| pair | anchor | `r_imit(3)` | `r_imit(20)` | ratio | `\|r_real/r_imit-1\|` at 3 / 20 | min median `R^2` | `p_r` unchanged at `k=20` | spend at `k=20` |
|---|---|---|---|---|---|---|---|---|
| audited | TinyComma-1.8B | `0.8278` | `0.8570` | `1.0353` | `0.0034` / `0.0006` | `0.9776` | `0.99947` | `171.3` |
| A | Llama-3.2-1B (base) | `0.6157` | `0.6174` | `1.0028` | `0.0005` / `0.0003` | `0.9908` | `0.99953` | `123.4` |
| B | Llama-3.2-3B-Instruct | `0.3010` | `0.3034` | `1.0080` | `0.0013` / `0.0007` | `0.9743` | `0.99954` | `60.6` |

### I1 — **SATURATES** at all three

`r_imit(20)/r_imit(3)` is `1.0353`, `1.0028` and `1.0080`, all inside the committed `1.10`. Raising
the budget from three nats per token to twenty moves the rate by at most `3.5\%`.

### I2 — **SPEND IS IMITATION** at all three

`|r_real/r_imit - 1|` is at most `0.0034` at `k=3` and `0.0007` at `k=20`, two orders of magnitude
inside the committed `0.05`. Above the pair's own imitation rate the decoder is not spending its
allowance on anything: it is paying to imitate, and the rest of the certificate is unreachable.

### I3 — **LINEAR** at all three

The median within-trajectory `R^2` of cumulative spend on the step index is at worst `0.9743`,
against a committed `0.95`, at every budget of every pair.

### I4 — **AS PREDICTED** at all three

At `k=20` the decoder serves `p_r` unchanged at `99.947\%`, `99.953\%` and `99.954\%` of steps.

### I5 — the prediction holds, **narrowly**, and the ordering is the interesting part

The band said pair B's `r_imit(3)` would be below **half** pair A's, because B's two models are the
same family and both instruction-tuned. Half of A's `0.6157` is `0.3079`; B reads `0.3010`. It
passes by `0.0069`, which is `2.2\%` of the threshold --- a pass, and reported as a narrow one.

What is not narrow is the ordering. Across the three pairs the rate falls monotonically with how
close the two models are --- `0.8278` for a 1.8B anchor trained on a different corpus entirely,
`0.6157` for a same-family base model, `0.3010` for a same-family instruction-tuned model --- and
the realised spend at `k=20` follows it exactly: `171.3`, `123.4`, `60.6` nats. The rate is
measuring the distance between the two distributions, which is what
`results/imitation_cost.csv` is read as in the paper and what I5 existed to be able to doubt.

### What this does and does not establish

Proposition 3 is a theorem and none of this confirms it; this file said so before the run. What is
established is that its *measured shape* --- saturation, a realised spend that converges on the
rate, linearity in the step index, and a decoder that ends as the risky model --- is a property of
the mechanism and not of the audited pair, at three pairs spanning a `2.8\times` range in the rate.

Neither Llama-3.2 anchor is a legitimate safe model, no near-access-freeness claim is made for
them, and excluded alternative 2 forbids quoting a certificate or leakage number from either.
Neither arm decoded anything on the protected split.
