# Pre-registration: the headline paired difference, every sampled arm re-drawn on disjoint seeds (feat-188)

**feat-188.** Committed **2026-09-24**, before any trajectory of this arm was generated. Nothing above
`## Scoring log` is edited afterwards.

## Why

Review 3 (Q12): is there a seed-level replication of the headline paired difference? The headline,
de-echoed (feat-184 A1, `results/order_averaged_h2h_deecho.csv`), is selection at `n=64` gaining
`+0.1015 [+0.0765, +0.1260]` over its own anchor and the metered decoder at `k=10` gaining
`+0.0510 [+0.025, +0.076]`, a paired difference `D3 = +0.0505 [+0.0155, +0.0860]`. Every arm in it was
drawn once, on the default seeds `42 43 44`. This re-draws **everything that is sampled**.

Caution (ap)'s rule, fixed before the data exist: `D3` sits `1.43` interval half-widths from zero,
below the `1.7` at which KL3M-1.7B's paired difference failed its own replication and well below the
`2.1`--`2.4` at which two others survived. **The project's own rule therefore predicts that this
reading may not replicate**, and we register it anyway.

## What runs (host B, GPUs 3 and 4; `scripts/run_replic.sh`)

`h1.py` with **`--seeds 52 53 54`** in place of the default `42 43 44` --- `build_trajectory_seeds`
hashes the tuple, so every seed differs from every seed on record --- and every other flag the committed
arm used:

| arm | on record | re-drawn as |
|---|---|---|
| anchor pool, `64` per prompt | `output/phase5/sel_anchor64`, batch `64` | `output/replic/pool_{neutral,creative,factual}`, batch `64`, merged into `output/replic/sel_anchor64` |
| metered `k=10` | `output/phase2/conc_all` (`kl_sweep_k10`), lowest seed | `output/replic/conc_k10`, `1` per prompt, batch `48` |
| anchor alone `k=0` | `output/sweep_plain`, lowest seed, batch `48` | `output/replic/anchor_k0`, batch `48` |
| opponent `k=-1` | `output/sweep_plain`, lowest seed, batch `48` | `output/replic/opp`, batch `48` |

The pool is generated one class per process; `h1.py` runs each class as its own sequence of seed groups,
so this does not change any batch's composition. The batch size of `kl_sweep_k10` is not in its log;
`48` is `run_regime_sweep.sh`'s, the launcher that wrote the arms beside it, and is stated as inferred.

**One difference from the arms on record that cannot be avoided:** the harness's left-pad slicing
defect was fixed on 2026-09-24 (caution (bc)), so the new pool's generations carry no prompt tail and the
reward model scores clean text, where the committed picks were made on echo-carrying text. The judge
reads recovered text in both. If the difference moves, this is the first candidate cause, and it is
stated beside the reading.

Reward: `selection_scaling.py --rewards-only`, the committed Qwen2.5-7B-Instruct template, batch `8`
as on record -> `results/selection_rewards64_replic.csv`. Judge: `order_averaged_h2h.py --deecho`,
judge B, seed `7717`, both presentation orders, `n=64`, `k=10`:

- **R1** against the **committed** opponent (`output/sweep_plain`): isolates the arms' own sampling.
  Tag `replic`.
- **R2** against the **re-drawn** opponent (`output/replic/opp`): the full replication. Tag `replic_opp`.

## Gates

- **G0.** Every arm covers the headline's `500` prompts, the pool has `64` draws on each, and no seed in
  any new arm appears in the arm it replicates.
- **G1.** The new pool's `n=1` empty fraction per class against the committed pool's, by two-proportion
  `z` test per stratum and on the total, `|z| < 2.58` (caution (v): a rate gate must be scale-free and
  stratified). Failing G1 makes the arm INVALID, not a failed replication (caution (w)).

## Bands

Readings of `D3`, R2 primary: **REPLICATES** if its interval excludes zero on the same side as
`+0.0505` (`lo95 > 0`); **REVERSES** if `hi95 < 0`; **DOES NOT REPLICATE** otherwise.

- **P1, R2's `D3`.** Predict **REPLICATES**, with the rule above saying the probability is not high:
  a fresh draw of the same size lands above its own interval's lower edge about `70%` of the time if
  the effect is exactly `+0.0505`, and less if the estimate on record is inflated.
- **P2, R1's `D3`.** Predict **REPLICATES**.
- **P3, `D1` (selection's own gain, `8` half-widths from zero).** Predict **REPLICATES** in both.

## What the manuscript does with each outcome, fixed now

REPLICATES: the headline sentence gains "and a draw on disjoint seeds gives `+x [lo, hi]`".
DOES NOT REPLICATE: the headline difference is reported with both draws, the second first, and every
sentence that rests on it is scoped to what both support; the two draws are **never pooled**
(caution (ap)). REVERSES: the same, and the abstract's head-to-head sentence is rewritten.

## Excluded in advance

- Pooling the two draws; choosing R1 or R2 after reading either.
- Any change of judge, opponent rank, seed, `n`, `k` or reward template after a judge call.

## Scoring log
