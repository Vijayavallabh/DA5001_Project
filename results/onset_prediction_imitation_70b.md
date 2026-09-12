# Pre-registration: Proposition 3 at the pair the mechanism's authors evaluated

Committed **before the run**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists

Every spend measurement in this paper uses `meta-llama/Llama-3.1-8B-Instruct` as the risky model.
He et al. evaluate anchored decoding with **TinyComma-1.8B against Llama 3.1 70B base** and sweep
`k` from `0.1` to `20` (AGENTS, Known truths). A reviewer is entitled to ask whether the paper's
central spend claim --- that the realised budget is the cost of *imitating* the risky model,
`Theta(T)` nats independent of the utility bought --- is a property of the mechanism or of the
smaller risky model we happened to audit it with.

The pair is runnable: the 70B base is in `hf_cache/` as the ungated mirror
`unsloth/Meta-Llama-3.1-70B`, it fits in bf16 across two A100s with TinyComma resident, and it
shares Llama-3.1's `128{,}256`-token vocabulary with the anchor. This is therefore the strongest
available version of Proposition 3's measurement: the mechanism's own published pair, at the
authors' own budgets.

Proposition 3 is a theorem and no measurement can refute it. What is at issue is whether the
*shape* the 1.8B/8B pair exhibits --- saturation, a realised spend that converges on the imitation
rate, a spend linear in the step index, a decoder that ends as the risky model --- survives a risky
model an order of magnitude larger, and whether the rate moves the way a distance between two
distributions should.

## What is measured

`k \in \{-1, 0, 0.5, 1, 3, 20\}` on the same `500` ordinary prompts (200 neutral, 150 creative, 150
factual) at the same temperature and `200`-token cap as every other arm, one trajectory each, with
the mandatory `k=-1` and `k=0` baselines on the same prompts and seeds. Zero-GPU analysis
afterwards from the per-step logs, exactly as for the 8B pair and the two Llama-3.2 pairs.

Nothing is decoded on the protected split and no leakage, certificate or `s(x)` number comes from
this arm --- those already exist at this pair from phase 2 and are not re-measured here.

## Bands, committed before the run

On record for the audited 8B pair, ordinary prompts: `r_imit` `0.3593` at `k=0.5`, `0.6654` at
`k=1`, `0.8278` at `k=3`, `0.8570` at `k=20`; realised spend `165.0` nats at `k=3` and `171.3` at
`k=20`; median within-trajectory `R^2` of cumulative spend on step index `>= 0.99` at every budget;
`p_r` served unchanged at `99.95\%` of steps at `k=20`. For the Llama-3.2-1B pair added today,
`r_imit(3) = 0.6157` and `r_imit(20) = 0.6174`.

**J1 -- does the imitation rate move with the risky model's distance from the anchor?** A 70B base
model is further from a 1.8B anchor than an 8B instruct model is, so its imitation rate should be
**higher**.

| reading | band |
|---|---|
| RATE MEASURES DISTANCE | `r_imit(3)` at the 70B pair exceeds the 8B pair's `0.8278` |
| FLAT | it is within `0.05` of it |
| INVERTED | it is more than `0.05` below, and the paper's reading of `imitation_cost.csv` as a distance is wrong |

**J2 -- does the shape survive?** The same four structural readings the two Llama-3.2 pairs were
scored on:

* SATURATES if `r_imit(20)/r_imit(3) <= 1.10`;
* SPEND IS IMITATION if `|r_real/r_imit - 1| < 0.05` at both `k=3` and `k=20`;
* LINEAR if the median within-trajectory `R^2` is `>= 0.95` at every budget;
* AS PREDICTED if `p_r` is served unchanged at `>= 0.99` of steps at `k=20`.

All four must hold to read SHAPE SURVIVES. Any that fails is reported as failing, and the paper's
Proposition 3 paragraph is qualified to the pairs where the shape holds.

**J3 -- the certificate's own arithmetic at this pair.** Realised spend at `k=3` against the
published cap `K = 3 T_{\max} = 600`. Reported with the spend/cap ratio; no band, because the
ratio's value follows from J1 and J2 rather than being an independent prediction.

## Excluded alternatives (named now so they cannot be adopted afterwards)

1. Changing the budget grid, the workload, the cap or the prompt set after seeing any number, or
   comparing against a differently generated 8B arm than `output/sweep_plain`.
2. Quoting a leakage, certificate or `s(x)` number from this arm. It decodes nothing protected.
3. Reporting J1 without J2, or either without the mandatory `k=-1` and `k=0` baselines.
4. Reading a confirmation of Proposition 3 into any of it. The proposition is proved; only the
   generality of its measured shape is at issue.
5. Dropping a budget because its arm is slow or its trajectories are few.
6. Using an aggregate counter where the per-step log is the source of truth (caution (l)).

## Scoring

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,4 HF_HUB_OFFLINE=1 \
  HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python h1.py --k-values -1 0 0.5 1 3 20 \
    --safe-model-path jacquelinehe/tinycomma-1.8b-llama3-tokenizer \
    --risky-model-path unsloth/Meta-Llama-3.1-70B --parallelize \
    --trajectories-per-prompt 1 --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
    --cap-val 0 --cap-test 0 --cap-attack-train 0 --max-new-tokens 200 \
    --output-dir output/phase5/imit_llama70b
.venv/bin/python analysis/imitation_cost.py --dirs output/phase5/imit_llama70b \
    --tag _llama70b --min-trajectories 20 --out results
```

---

## Scoring log (appended after the run; nothing above this line is edited)

## Addendum, 2026-09-12 20:02, written while the arm is still generating and before any number exists

J1 has a confound I did not name when I wrote it, and naming it after the result would be worthless.

The band reads "a 70B base model is further from a 1.8B anchor than an 8B instruct model is, so its
imitation rate should be **higher**". That treats size as the only axis of distance. It is not.
`Llama-3.1-8B-Instruct` is **instruction-tuned** and TinyComma is a base model, so the `0.8278` on
record already contains the base-to-instruct shift; `unsloth/Meta-Llama-3.1-70B` is a **base** model
and shares that much of its style with the anchor. Size pushes the 70B rate up and the missing
instruct shift pushes it down, and nothing on record says which wins.

Today's two Llama-3.2 pairs point the same way: `0.6157` for a same-family **base** anchor against
the 8B instruct, `0.3010` for a same-family **instruct** anchor against it. Both of those vary the
anchor rather than the risky model, so neither settles it, but they do show the instruct axis moving
the rate by more than the size axis does across `1`B to `3`B.

**So J1 as written is not a clean test of "the rate measures distance", and its INVERTED reading
must not be applied as stated.** Concretely, if the 70B rate comes back below `0.8278`:

* the reading is **not** that the distance interpretation is wrong;
* it is that base-versus-instruct is a larger axis than parameter count, which is itself worth
  reporting and is consistent with the three pairs already measured;
* the sentence in the appendix that says the rate "moves the way a distance should" must then name
  *which* distance --- style and tuning, not size --- or be cut.

J2 and J3 are unaffected: they are about the shape, not the level, and nothing above changes what
they measure. This addendum is committed before the arm finishes so the record shows it was not
written to accommodate a number.
