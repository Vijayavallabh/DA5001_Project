# Pre-registration: the head-to-head at the authors' decoding settings, temperature 0.7 and repetition penalty 1.1 (feat-195)

**feat-195.** Committed **2026-09-24**, before any token of the arms below is decoded. Nothing above
`## Scoring log` is edited after the first judge call.

## Why

Three referee reports (2026-09-24, sixth round) name the same validity threat: every judged
head-to-head runs the meter at temperature `1.0` with no repetition penalty, where He et al. decode
books at `0.7` and `1.1` (their App. D.1), and at `1.0` the `70`B base is itself judged no better
than the anchor (`+0.019 [-0.004, +0.042]`, feat-185). One report asks whether the sign survives at
the authors' settings, another whether the `70`B then beats the anchor, a third whether the meter
gains anything over its anchor at `k <= 0.5`. This arm answers all three with one change: decoding.

## What runs

`h1.py` gains `--repetition-penalty` (default `1.0`, so no run on record changes); the factory already
applies temperature and penalty to BOTH logit vectors before the solve (`a_patch/factory.py`, decode
loop), which is He et al.'s App. B. Every arm below decodes at `--temperature 0.7
--repetition-penalty 1.1`, on the headline's `500` prompts, `T_max = 200`, one trajectory per prompt
at the default seeds, and the selection pool's certificate is relative to the same warped anchor.

- `output/feat195/t07_8b`: TinyComma with `Llama-3.1-8B-Instruct` served without its chat template
  (the released logs' configuration), `k in {-1, 0, 0.5, 1, 10}`, batch `48`, GPU 4.
- `output/feat195/t07_70b`: TinyComma with `unsloth/Meta-Llama-3.1-70B` (the authors' token-level
  pair), `k in {-1, 0.5, 1, 20}`, `--parallelize` over GPUs 1 and 2, batch `8` (the committed
  70B run's).
- `output/feat195/t07_pool64`: the selection pool, `k = 0` with `64` trajectories per prompt,
  self-paired on the anchor (`--risky-model-path` = the anchor). At `k = 0` the harness serves the
  anchor alone whatever the risky model is, so self-pairing changes no draw's law and removes an
  `8`B forward pass nobody uses; it is the cost table's path B. Batch `256`.
- Rewards: `Qwen2.5-7B-Instruct`, the committed template and `analysis/compute_matched.reward_cache`
  path, over all `32,000` candidates -> `results/selection_rewards64_t07.csv`.

Judging: `analysis/order_averaged_h2h.py --deecho`, judge B, seed `7717`, both orders. The opponent
is the `8`B-Instruct's own draw at the SAME settings (`t07_8b`, `k = -1`), as Table 3's two
text-continuation blocks share the templateless `8`B opponent; the meters' control is the anchor alone
at the same settings (`t07_8b`, `k = 0`), which is the law every meter here tilts away from.

- **J1** metered `8`B `k=10`, extra `8`B `k=0.5`. **J2** metered `8`B `k=1`, extra the `70`B alone.
- **J3** metered `70`B `k=20`, extra `70`B `k=0.5`. **J4** metered `70`B `k=1`.

## Gates

- **G0.** Every arm covers the `500` prompts; the pool holds `64` draws per prompt and the reward
  cache `32,000` rows.
- **G1.** The settings took effect: every record carries `temperature = 0.7` and
  `repetition_penalty = 1.1`, and the `k = -1` text differs from the committed temperature-`1.0`
  opponent's on at least `99%` of prompts.
- **G2.** Selection's per-prompt levels are identical across J1 to J4 (same text, same opponent,
  greedy judge).

## Predictions, on the script's own readings

`D3` is selection's gain over its own control minus the meter's over its own, paired; the script
reads CONFIRMED (interval above zero), REFUTED (below) or UNRESOLVED. `D5` is an extra arm's gain
minus selection's: INCUMBENT WINS / INCUMBENT LOSES / TIE.

- **H1, the headline cell, `8`B at `k=10`.** We predict **UNRESOLVED or CONFIRMED** and do not
  predict the sign: a lower temperature makes the risky model a better continuer, and it also makes
  the anchor's draws more coherent and less diverse, which is what best-of-`n` spends.
- **H2, the one budget whose certificate says anything about a `50`-token window, `8`B at `k=0.5`.**
  We predict **INCUMBENT LOSES**: where the budget binds the meter is its anchor, as at `1.0`.
- **H3, the authors' pair at a vacuous budget, `70`B at `k=20`.** We predict **REFUTED**: at `0.7` a
  `70`B base continuing text should be a far better text model than best-of-`64` over a `1.8`B
  anchor. feat-185 predicted the same at `1.0` and was wrong, which is why this is re-asked here.
- **H4, descriptive.** The `70`B alone gains over the anchor with an interval above zero.
- **H5, descriptive.** `70`B at `k=0.5`: its `D5` against selection, read as for H2.

## What the manuscript does with each outcome, fixed now

The four passes become a temperature-`0.7` block of Table 3, stated as the authors' decoding settings.
Limitations stops saying the meter ran only at `1.0`. If H1 or H3 reads REFUTED, the abstract and
Section 4 scope every "at every budget" claim to the configuration it was measured in and state the
`0.7` result beside it, in the same sentence; if they read CONFIRMED the claim is stated at both
temperatures. Whatever H2 and H5 read, the sentence about the informative budget reports them.

## Excluded in advance

- Any other temperature, penalty, seed, judge or budget chosen after a judge call.
- Pooling with the temperature-`1.0` passes, or setting a level from these passes against one from
  another pass; only paired differences within a pass are quoted.
- Reading `K/S_w` at `0.7` off the `1.0` anchor: the certificate is relative to the warped anchor.

## Compute

About two hours of GPUs 1 and 2 for the `70`B, and three of GPU 4 for the `8`B arms, the pool and
the rewards; four judge passes of about six minutes each.

## Addendum before launch, 2026-09-24 23:15 IST (no token decoded, no judge called)

- **Host.** The first launch (22:05 IST, local) died on a CUDA allocation in every job and wrote no
  trajectory (`output/logs/feat195_failed_launch_2209/`, `output/feat195/` empty): minutes earlier
  another job of this account took local GPUs 0, 1, 2 and 4, and they are still held. The arms run on
  host B's idle H100s instead: the `8`B arms, the pool and the rewards on GPU 4, the `70`B on GPUs 5
  and 6, J1/J2 on GPU 7 and J3/J4 on GPU 5 (`GPU_MAIN=4 GPUS_70B=5,6 GPU_J12=7 GPU_J34=5`). Every arm
  and all four judge passes run on this one host, so each paired difference is within one host's
  arithmetic (caution (as)).
- **The `70`B's sharding.** The launcher omitted the committed `70`B run's `--risky-device-map auto
  --max-memory 0=75GiB,1=70GiB` (`output/phase5/imit_llama70b`), without which the factory pins the
  `70`B to one card (caution (q)); the first launch's `70`B job died on exactly that allocation
  (`78.05` GiB on one card). Restored, as "the committed `70`B run's" settings above already promise.
- Nothing else changes: prompts, seeds, budgets, batch sizes, judge, gates and predictions are as
  written above.

## Scoring log

### Scored 2026-09-25 00:24 IST --- H1 REFUTED (wrong), H2 TIE (wrong), H3 CONFIRMED (wrong), H4 right, H5 INCUMBENT LOSES

All jobs exited `0` on host B (`output/logs/feat195_{t07_8b,t07_70b,pool,rewards,J1,J2,J3,J4}.done`,
launched 23:00 IST, J4 last at 00:20). Scored by `.venv/bin/python analysis/he_decoding.py --out results`
-> `results/he_decoding.csv`; Table 2's rows by `analysis/served_opponent.py` (rows `temperature 0.7`,
appended, so no committed row's bootstrap draw moved: the first `73` rows are byte-identical) and
`analysis/served_activity.py` (rows `T=0.7`).

**Gates.** G0 PASS: `500` prompts in every pass; the pool holds `32,000` trajectories (`64` per prompt,
all within budget) and the reward cache `32,000` finite rows. G1 PASS: every `k=-1` record carries
`temperature = 0.7` and `repetition_penalty = 1.1`, and its text differs from the committed
temperature-`1.0` opponent's on `500/500` prompts (read by the judge's own loader). G2 PASS: selection's
per-prompt levels are identical in J1-J4. Every class of both new sweeps is within budget with `0`
invariant violations.

**`K/S_w` against the warped anchor, as registered.** `analysis/regimes.py` gained `--temperature` and
`--repetition-penalty` (the penalty is applied before the temperature, as the decoder's processor runs
before its warper; checked equal to HF's `RepetitionPenaltyLogitsProcessor` to `0.0`). A same-host control
at `1.0` reproduces the committed median `3.1966` nats per token as `3.1950` (`S_w = 159.8` both), and the
warped anchor reads `3.5479`, so `S_w = 177.4` nats (`results/regimes_copybench_t07.csv`). `K/S_w` is
`0.56`, `1.13`, `11.3` and `22.5` at `k = 0.5, 1, 10, 20`.

| band | reading | registered | verdict |
|---|---|---|---|
| H1, `8`B `k=10`, `D3` | `-0.081 [-0.113, -0.049]`, REFUTED | UNRESOLVED or CONFIRMED | **wrong** |
| H2, `8`B `k=0.5`, `D5` (meter minus selection) | `-0.0245 [-0.057, +0.0095]`, TIE | INCUMBENT LOSES | **wrong** |
| H3, `70`B `k=20`, `D3` | `+0.0485 [+0.0145, +0.082]`, CONFIRMED | REFUTED | **wrong** |
| H4, the `70`B alone over the anchor | `+0.0585 [+0.034, +0.0835]` | interval above zero | right |
| H5, `70`B `k=0.5`, `D5` | `-0.089 [-0.122, -0.0575]`, INCUMBENT LOSES | read as H2 | --- |

Descriptive: `8`B `k=1` `D3 = -0.0225 [-0.056, +0.0115]`; `70`B `k=1` `D3 = +0.100 [+0.067, +0.1335]`;
selection's gain `+0.107 [+0.0825, +0.132]`; the `k=0.5` meters over their anchor: `8`B `+0.0825
[+0.0605, +0.1045]`, `70`B `+0.018 [-0.0035, +0.040]`. Active shares: `8`B `44.6%`, `16.4%`, `0.040%`,
`70`B `46.4%`, `20.2%`, `0.00%` at `k = 0.5, 1, 10|20`.

**What it means.** At the authors' decoding settings the `8`B-Instruct continuing text beats selection
at the vacuous budget, and at the one budget that certifies a window the `8`B meter blends on `45%` of
steps, gains over its anchor and ties selection. The authors' own `70`B base, a weaker continuer under
this judge, still loses to selection at every budget.

**Manuscript, as registered.** Table 2 gains the temperature-`0.7` block (`S_w = 177.4` under the warped
anchor in its caption). H1 reads REFUTED, so the abstract and Section 4 scope the continuing-text claim
to temperature `1.0` and state the `0.7` result in the same sentence; Section 4's sentence about the
informative budget reports H2 and H5; Limitations no longer says the meter ran only at `1.0` and lists
the authors' settings among the configurations that reverse the comparison. Beyond the registration,
the Section 4 heading and the contributions list stop saying the meter "is its anchor" wherever its
certificate says anything: at `0.7` the `8`B meter is not.
