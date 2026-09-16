# The four Rényi orders' missing baselines, and what they must equal

**Committed 2026-09-16 09:15, before the runs exist.** Nothing above the `## Scoring log` heading is
edited after that line is written.

## The gap

`output/phase4/renyi_renyi_{1_0,2,4,8}` each sweep `k ∈ {1, 3, 5}` on `single` and `oracle` `L=50`
and carry **no `k=-1` and no `k=0` arm**. They feed Table 3, the four-order comparison. Working
Rules require every experiment reporting a copying metric at some `k` to report both baselines on the
same prompts and seeds; these four do not. Two phase-4 sweeps had the same defect and were measured
earlier today, so this closes the last of the six real cases a scan of every sweep on disk found.

## The prediction, which is a strong one

At `k=-1` the decoder serves the risky model alone and at `k=0` the safe model alone. Neither is a
constrained decode, and `a_patch/factory.py` says so in its structure: the `k_radius == 0.0` and
`k_radius == -1.0` branches build their mixing weights directly and **never consult
`self.constraint`**, which is read only in the `else` branch that solves under a budget.

> **All four orders' `k=-1` and `k=0` arms will be identical to each other, and identical to
> `output/phase4/fine_tc_base`, to every decimal the CSV stores** --- on `single` and on `oracle
> L=50`. The pair is the same (TinyComma-1.8B anchor, memorising Llama-3.1-8B), the corpus is the
> same 100 passages, and the protocol lines differ only in `constraint=`.

Concretely, from `fine_tc_base` measured at 08:40 today:

| arm | nv-recall | lcs\_word |
|---|---|---|
| `k=-1` single | `0.4921` | `81.34` |
| `k=-1` oracle L=50 | `0.8074` | `107.35` |
| `k=0` single | `0.0000` | `1.73` |
| `k=0` oracle L=50 | `0.0032` | `3.79` |

## What a failure would mean, written down before the numbers exist

**Any** difference refutes it, and the consequence is larger than this arm. It would mean the
constraint reaches a decode that is supposed to be unconstrained --- either directly, or through the
RNG, since these arms are *sampled* and a constraint object that consumes the stream differently
would move the draw without touching the mixing weights. Every `k=-1` and `k=0` baseline in this
paper would then be constraint-specific rather than a property of the models, and Table 3's four
orders could not be read against a shared baseline. It would be investigated before anything is
quoted, not patched around.

A difference confined to `oracle` would be narrower: oracle re-samples per window, so it exercises
the RNG far more than `single` does, and a discrepancy there with `single` intact would point at the
stream rather than at the mixing weights.

## Protocol

Every flag at its default except the ones that reproduce each sweep's own protocol line: the models,
`--k-values -1 0`, `--modes single oracle`, `--windows 50`, `--limit 100`, and
`--constraint renyi:<alpha>` for each of the four orders. **`--batch-size` is left at its default of
32**: it is part of the seed for a sampled arm (caution (u)) and passing one explicitly is a change.

**What will not happen.** No re-run at a different batch size or seed, no dropping an order that
disagrees, and no substituting `fine_tc_base`'s numbers for an order that fails to reproduce them.

## Scoring log
