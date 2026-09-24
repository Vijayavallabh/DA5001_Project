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

## Scoring log
