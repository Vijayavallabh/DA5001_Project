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

`h1.py` with **`--seeds 82 83 84`** in place of the default `42 43 44` --- `build_trajectory_seeds`
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

*Corrected 2026-09-24 14:25 IST, before any reading:* this registration first named `--seeds 52 53
54`. That tuple is **not** new: `output/phase5/sel_anchor64_seed52` already re-draws the headline pool
on it (Appendix table `tab:h2hrepeat`, `D3 = +0.0635` on echo-carrying text, meter not re-drawn), and
feat-184's chat arm used it too, so the claim above that every seed is new was false. About `35`
minutes of pool generation on that tuple were stopped and moved aside
(`output/replic_seed52_aborted_2026-09-24`, never read). `82 83 84` appears nowhere in the repository.
The existing seed-52 re-draw is added as a descriptive arm:

- **R0** the seed-52 pool on record, re-judged on recovered text against the committed meter, anchor
  and opponent (`--sel-dir output/phase5/sel_anchor64_seed52 --rewards
  results/selection_rewards64_seed52.csv`, tag `seed52_deecho`), so the replication already in the
  paper is read under the same text repair as the headline.

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

*Added 2026-09-24 14:46 IST, before the new pool was complete or scored:* **two secondary readings.**
The hybrid arm (feat-189, generated on the fixed harness) showed the committed reward rating an
**empty** completion above a typical anchor completion --- median log-odds `-12.1` against `-26.8`
over its `4,000` drafts --- so best-of-`n` serves an empty draft whenever one exists. Every pool on
record masked this, because its empty draws carried the prompt's tail (caution (bc)); this
replication's pool is the first on record recorded correctly, so its primary readings can move for a
reason that has nothing to do with the headline. **R1b** and **R2b** repeat R1 and R2 with empty
candidates excluded from the argmax (`analysis/nonempty_rewards.py`: reward `-1e9` where
`n_words == 0`, so an empty draw is served only if all `64` are empty), same bands, **reported beside
the primaries and never in their place**; the primaries stay as registered.

## Excluded in advance

- Pooling the two draws; choosing R1 or R2 after reading either.
- Any change of judge, opponent rank, seed, `n`, `k` or reward template after a judge call.

## Scoring log

### Scored 2026-09-24 16:00 IST --- P1, P2, P3 REPLICATE; G0 and G1 pass

Generation: the neutral pool and the three single-draw arms locally (GPUs 4 and 2, after the queue
was re-dealt across two cards; same commands), the creative and factual pools on host B; the reward
cache and every judge pass on host B (R1 on GPU 4, R1b, R2, R2b in parallel on GPUs 3, 5, 6, the
registered commands unchanged); R0 locally. `analysis/replic_score.py --out results` ->
`results/headline_replication.csv`.

**G0 PASS**: every new arm covers the `500` prompts, the pool has `64` draws on each, and no prompt of
any arm shares a seed with the arm it replicates. **G1 PASS**: the new pool's `n=1` empty fraction
against the committed pool's (both on the true generations) is `34/200` vs `47/200` on the neutral
class (`z = -1.62`), `21/150` vs `17/150` on the creative (`z = 0.69`), `0/150` vs `0/150` on the
factual, and `55/500` vs `64/500` in total (`z = -0.88`), all inside `|z| < 2.58`.

| run | opponent | `D1` selection | `D2` meter | `D3` | band | reading |
|---|---|---|---|---|---|---|
| **R2** | re-drawn | `+0.1225 [+0.0990, +0.1460]` | `+0.0730` | `+0.0495 [+0.0175, +0.0800]` | P1 | **REPLICATES** |
| **R1** | committed | `+0.1065 [+0.0855, +0.1280]` | `+0.0615` | `+0.0450 [+0.0110, +0.0790]` | P2 | **REPLICATES** |
| R1b | committed, non-empty argmax | `+0.1235 [+0.1020, +0.1455]` | | `+0.0620 [+0.0275, +0.0960]` | | REPLICATES |
| R2b | re-drawn, non-empty argmax | `+0.1425 [+0.1200, +0.1655]` | | `+0.0695 [+0.0385, +0.0995]` | | REPLICATES |
| R0 | committed; seed-52 pool, repaired text | `+0.1010 [+0.0765, +0.1245]` | `+0.0510` | `+0.0500 [+0.0155, +0.0840]` | | descriptive |

`D1` REPLICATES in R1 and R2 (P3). Every prediction was right. Against the committed `+0.0505
[+0.0155, +0.0860]`, three independent draws now read `+0.0495`, `+0.0450` and `+0.0500` --- the first
with every sampled arm re-drawn, opponent included --- and none is pooled with another (caution (ap)).
Excluding empty candidates from the argmax raises the difference (`+0.0620`, `+0.0695`): the committed
reward's preference for empty text (`results/empty_preference.csv`) costs selection level here rather
than flattering it. **What the manuscript does, as fixed above:** the headline sentence gains "and a
draw on disjoint seeds gives `+0.0495 [+0.0175, +0.0800]`", and the repeat table gains the rows.
