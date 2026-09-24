# Pre-registration: what one request costs when a server batches the n draws (feat-190)

**feat-190.** Committed **2026-09-24**, before any timed cell (one 20-token smoke of the selection cell
checked the script end to end and is not a reading). Nothing above `## Scoring log` is edited afterwards.

## Why

Review 4 (B.3): win on compute, "batched or prefix-shared anchor sampling vs the meter, especially at
the 70B pair". Every price on record charges `n` as `n` sequential decodes: the harness realises `n`
as `n` seed groups (caution (ay)), and feat-164's deployable cell C timed the anchor with **one**
completion per prompt, so `21.8x` is `64 x 4.463 / 13.074` --- sixty-four decodes of width `W`. A
server that batches a request's `64` candidates runs **one** decode loop of width `64W` over a shared
prompt, and decoding a `1.76`B model at small width is bound by weight and cache reads, not FLOPs. And
selection never runs the risky model, where the metered decoder runs it at every step --- a 70B at the
authors' own pair.

## Cells (`analysis/batched_latency.py`)

Every cell serves exactly `T = 200` new tokens per completion (`min_new_tokens`), temperature `1`, the
headline's neutral prompts, rep `r` on prompts `[rW, (r+1)W)`, `3` reps, models loaded once and the
load timed apart, one untimed warm-up. Widths `W in {1, 8}` prompts per call.

- **SEL**: TinyComma, one `generate(num_return_sequences=n)` for `n in {1, 8, 64}`, then **one**
  reward pass over the `Wn` candidates with the committed scorer (`selection_scaling.score_rewards`,
  Qwen2.5-7B-Instruct, batch `64`), both on one card.
- **MET**: `a_patch` `AnchoredDecodingFactory.generate` at `k = 10`, prefix debt on, both models every
  step as the mechanism requires; the 8B pair on one card, the 70B pair on the cards it needs
  (TinyComma beside the 70B's shards), `--parallelize` where the two models sit on different cards.
- **RISKY**: the risky model alone, the undefended server, as a reference.

Per-request seconds = (generation + reward) / `W`. The box is **shared** (the other project's jobs
hold cards on both hosts, caution (ay)'s idle-box rule cannot be met before the deadline), so every
cell reports its repeat spread, arms are timed back to back on the same card, and a ratio is read only
where it exceeds its cells' spreads.

## Bands (per-request seconds, `W = 1`)

- **T1, `SEL(n=64) / MET(70B pair)`.** Predict **below `1`**: the meter reads `141` GB of weights per
  token; selection reads `3.5` GB and a cache that sixty-four short sequences share a batch for.
- **T2, `SEL(n=64) / MET(8B pair)`.** Predict **between `0.5` and `2`**.
- **T3, `SEL(n=64) / SEL(n=1)`.** Predict **below `8`**: batching `64` candidates costs far less than
  `64` decodes (the harness's own measurement is `64 x`, since it runs them in sequence).
- **T4, descriptive.** The same ratios at `W = 8`, where batching has less idle bandwidth to use and
  the ratios should rise.

## What the manuscript does with each outcome, fixed now

Selection's cost sentence (Section 3) and the Conclusion quote the batched per-request ratio at both
pairs beside the unbatched `21.8x`, which stays as the throughput-bound price. If T1 holds, the paper
says that at the authors' own pair selection serves a request faster than the meter. The matched-
compute concession is about FLOPs and is **not** revised by a latency number.

## Excluded in advance

- Dropping a slow repeat; choosing `W`, `n`, `k` or the reward batch after seeing a timing.

## Scoring log

### Scored 2026-09-24 15:50 IST --- T1 CONFIRMED (`0.220`), T3 CONFIRMED (`1.28`), T2 on its band edge within noise (`0.514`); at eight requests per call every ratio rises, as T4 expected

Host B (H100s, a box shared with the other project's jobs): the single-card part on GPU 7
(11:07--11:12 CEST: SEL, MET 8B, RISKY 8B, SEL again), the 70B part on GPUs 3, 5, 6 with its SEL cell
on GPU 3 (12:02--12:08). `analysis/batched_latency.py --report --logs
output/logs/batched_latency_hostb_all.log` -> `results/batched_latency.csv` (every cell with its repeat
spread) and `results/batched_latency_bands.csv`. **Ratios are formed within one part only**, so each
selection cell is compared with the meter it was timed beside on the same card; the first version of
the report pooled both parts' selection cells, which the same-card rule above does not allow, and was
changed before any band was read.

| per-request seconds | `W = 1` | `W = 8` |
|---|---|---|
| selection `n=1` / `8` / `64` (single part) | `2.41` / `2.49` / `3.09` | `0.336` / `0.412` / `1.46` |
| selection `n=64` (70B part, GPU 3) | `3.27` | `1.46` |
| meter, `8`B pair, `k=10` | `6.00` | `0.866` |
| meter, `70`B pair, `k=10` (GPUs 3, 5, 6) | `14.84` | `2.09` |
| `8`B alone / `70`B alone | `3.19` / `12.19` | `0.462` / `1.74` |

| band | ratio | cell spreads | predicted | reading |
|---|---|---|---|---|
| T1 `SEL(64) / MET(70B)`, `W=1` | `0.220` | `0.027` | below `1` | **CONFIRMED** |
| T2 `SEL(64) / MET(8B)`, `W=1` | `0.514` | `0.048` | in `[0.5, 2]` | **WITHIN NOISE** of the lower edge (`2.9%` from it, spreads `4.8%`); not read |
| T3 `SEL(64) / SEL(1)`, `W=1` | `1.28` | `0.065` | below `8` | **CONFIRMED** |
| T4 at `W=8` | `0.696` (70B), `1.69` (8B), `4.35` (`n=1`) | | should rise | all three rose |

**What the manuscript does, as fixed above.** Section 3's cost sentence and the Conclusion quote the
batched per-request ratio at both pairs beside the unbatched `21.8x`, which stays as the
throughput-bound price; T1 held, so the paper says that at the authors' own `70`B pair selection serves
one request in about a fifth of the meter's time --- `0.70x` still at eight requests per call --- and
that at the `8`B pair it is about half the meter's time for one request but `1.69x` at eight. Selection
at `n=64` also serves a request in `0.27x` the time of the `70`B model alone. Nothing here revises the
matched-compute concession, which is about FLOPs.
