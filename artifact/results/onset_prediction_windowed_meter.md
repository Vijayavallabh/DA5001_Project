# Pre-registration: the sliding-window meter, and every mechanism at a matched certificate (feat-210)

**feat-210.** Committed **2026-09-25**, before any token of these arms is decoded and before any judge call
of the matched pass. Nothing above `## Scoring log` is edited after the first token.

## Why

Seventh review round. All three reviews ask for the design Section 3 leaves open and argues about without
running: a meter that bounds every `w`-token window instead of the whole sequence (review 1 Q2, "at window
budgets of `log 64`, 25, 40 and 125 nats"; review 2 W7/Q2; review 3 Q8 and roadmap B1). Review 3 (A2, Q1)
also asks that the comparisons at a MATCHED certificate be primary, run in one pass: selection, installments
and every meter at the same order and the same number of nats, with the vacuous `k=10` comparison secondary.
Review 3 (A3, Q2, Q3) asks for the instrument's validity: a judge with order consistency above `0.7`, the
headline restricted to order-consistent items, a known quality difference the judge must detect, and the
power of the `k=0.5` null.

## The mechanism

`a_patch/factory.py`, `window=w` (tests: `tests/test_windowed_meter.py`, a real decode on two tiny models).
A pathwise meter whose allowance at step `t` is `W` minus the largest realised log-ratio of any span of the
last `w-1` served tokens that ends at `t-1` (never below `0`; negative ratios earn no credit). So on every
path every span of at most `w` output tokens has realised log-ratio at most `W`, and any event on such a
span given its prefix has `q(E) <= e^W p_s(E)`: an exact protected `50`-token window of anchor surprisal
`S_w` is certified at `e^{W - S_w}`. No prefix debt: the output windows are the certified object. `w = 50`.

## What runs (host B, GPUs 4-7; launcher `scripts/run_feat210.sh`)

Common: `h1.py --trajectories-per-prompt 1 --seeds 52 53 54 --cap-neutral 200 --cap-factual 150
--cap-creative 150 --cap-val 0 --cap-test 0 --cap-attack-train 0 --batch-size 48 --trust-remote-code`, the
headline's `500` prompts, TinyComma-1.8B with Llama-3.1-8B-Instruct, temperature `1.0`, no penalty,
`T_max = 200`; seeds disjoint from the opponent's `42 43 44`.

- **win_plain**: `--constraint pathwise --window 50 --no-prefix-debt --k-values 0 4.1589 12.4767 24.9534 40
  125` -> `output/feat210/win_plain`. `4.1589 = log 64` (selection's certificate); `24.9534` and `12.4767`
  are installments' per-window certificates at `L = 10` and `25` (`(ceil(50/L)+1) log 64`); `40` and `125`
  are review 1's. The `k=0` arm is the anchor alone, these arms' own control.
- **pw_plain**: `--constraint pathwise --no-prefix-debt --k-values 0.415888 0.166355` -> `output/feat210/pw_plain`,
  the per-token pathwise meter at the whole-output certificates of installments at `L = 10` (`83.178` nats)
  and `L = 25` (`33.271`).
- **front_plain**: `--constraint pathwise --no-prefix-debt --initial-bank 4.158883 --k-values 1e-9` ->
  `output/feat210/front_plain`, the whole budget `log 64` up front, pathwise.
- **win_chat**: `--use-chat-template`, otherwise win_plain's flags, `--k-values 0 12.4767 24.9534 40 125` ->
  `output/feat210/win_chat`.
- **leak_plain**: `analysis/composition_attack.py --risky-model output/memorizing_llama8b --split attack_train
  --limit 100 --modes single --constraint pathwise --window 50 --no-prefix-debt --k-values -1 0 4.1589 12.4767
  24.9534 40 60 80 100 125 150 200` (every other flag the script's default: seed `20` tokens, batch `32`,
  seed `1234`, temperature `1.0`) -> `output/feat210/leak_plain`.
- **leak_chat**: the same with `--use-chat-template --k-values -1 0 24.9534 40 80 125` -> `output/feat210/leak_chat`.

## Judging (local A100, one pass per judge)

`analysis/matched_h2h.py`: every arm against the committed opponent (`output/sweep_plain` `k=-1`, lowest
seed), both presentation orders, generations de-echoed, `1,200` characters, judge~B
(`microsoft/Phi-3.5-mini-instruct`, the pre-specified judge) and then judge~G (`google/gemma-2-27b-it`, the
judge with the highest measured order consistency, `0.784`-`0.846`). Each arm is judged in batches of `200`
in prompt order, as `order_averaged_h2h.py` does, and every verdict is saved (`results/matched_h2h_verdicts_<tag>.csv`).

Arms: the committed four (`sel_n64`, `sel_n1`, `metered_k10` from `output/phase2/conc_all`, `anchor_k0` from
`output/sweep_plain`); the non-empty rule (`output/feat209/pool`); installments `blk10n64`, `blk25n64` and
their own `n=1` pipeline arm `blk200n1` (`output/feat201`); the KL meter at `k=0.5` (`conc_all`); the KL
front-loaded meters at `log 64` and `64` nats (`output/phase5/sparse_b4.16_t0`, `sparse_b64_t0`); every new
plain arm; and a second, independent draw of the opponent's own configuration (`sweep_plain` `k=-1`,
second-lowest seed), the known quality difference.

**Chat pass**, judge~B and judge~G: `sel_n64`, `sel_n1`, the non-empty rule, `output/feat184/chat_k10` and
every win_chat arm, against `output/sweep_chat` `k=-1`, lowest seed.

**Gain.** Each arm's order-averaged level minus its own control's: selection-family arms against `sel_n1`;
installments against `blk200n1`; the committed meters against `anchor_k0`; every feat-210 arm against
win_plain's (win_chat's) `k=0`. **Difference.** Two arms' gains, paired over prompts, `10,000` resamples,
`95%`, labelled CONFIRMED (above zero) / UNRESOLVED / REFUTED (below).

## Gates

- **G0.** Every generation arm covers the `500` prompts; each leakage run the `100` passages.
- **G1.** Every windowed trajectory has its worst span of at most `50` tokens within `W + 1e-3`, recomputed
  from the per-step log; every per-token pathwise trajectory its total within `K + 1e-3`; the front-loaded
  one within `4.1589 + 1e-3`; every leakage query likewise. One violation voids its arm.
- **G2.** In judge~B's plain pass the four committed arms' per-prompt levels equal
  `results/order_averaged_h2h_per_prompt_deecho.csv` on `500` of `500` prompts, so this pass IS the
  committed instrument. If not, the pass stands alone and none of its levels is set beside a committed one.

## Predictions (plain pass, judge~B unless named)

- **P1** matched to selection's certificate, `log 64` pathwise per window and per output: selection's gain
  minus the windowed meter's at `W = 4.1589` and minus the front-loaded pathwise meter's: **CONFIRMED** both.
- **P2** selection minus the windowed meter at `W = 24.9534`, `40` and `125`: **CONFIRMED** at each (at `125`
  the meter serves the risky model almost everywhere, as the `k=10` meter does).
- **P3** matched whole-output certificate: installments minus the per-token pathwise meter, `L=10` against
  `K=83.178` and `L=25` against `K=33.271`: **CONFIRMED** both.
- **P4** matched per-window certificate: installments minus the windowed meter, `L=10` against `W=24.9534`
  and `L=25` against `W=12.4767`: **CONFIRMED** both.
- **P5** chat pass: selection minus the windowed meter at `W=125`: **REFUTED** (the meter wins, as the `k=10`
  chat meter does); at `W=12.4767`: **CONFIRMED**. `24.9534` and `40`: descriptive.
- **P6** leak_plain and leak_chat, each: mean near-verbatim recall **below `0.01`** at every `W <= 40` and
  **at least `0.01`** at `W = 125`, an extraction the exact-window certificate at `e^{125 - S_w}` does not
  exclude.
- **P7** judge~G: every difference in P1-P4 has the **same sign** as under judge~B.

**Descriptive, no band.** The windowed onset (the smallest grid `W` whose mean recall reaches `0.01`, plain
and chat); binding and forced shares per arm; each arm's realised worst span; installments against the KL
meter at `k=0.5` (matched total nats, different orders); the headline difference restricted to prompts whose
verdicts are order-consistent (both compared arms, and all four); the known quality difference (the second
opponent draw's level minus the anchor's) and the minimum difference the `k=0.5` null had `80%` power to
detect; the level of items whose served text is empty, per judge.

## What the manuscript does with each outcome, fixed now

The matched comparisons become Section 4's first table and Section 3's paragraph on designs the dichotomy
does not cover reports the windowed meter as measured; the `k=10` comparisons move to a second table headed
as comparisons against the risky model. If P2 or P5 fails, the windowed meter is stated as matching or
beating selection wherever it does, at the window certificate it does so with. If P5 holds and leak_chat reads
below `0.01` at `W = 125` (it wins under the chat template without leaking), the abstract and the
conclusion name the windowed meter as the mechanism for that regime. Wherever P6 holds, the paper states
that an exact-window certificate of `W` nats coexisted with near-verbatim extraction at that `W`.

## Excluded in advance

- Any other window length, budget, seed, temperature, template, judge or control chosen after a token is
  decoded or a verdict read; pooling this pass with any other.

## Scoring log

### Addendum 2026-09-25 19:05 IST, before it runs --- selection on the leakage sweep's own protocol (descriptive, no band)

Review 1 (Q6) asks for meter and selection leakage "on the same passages, the same target lengths, and the
same scorer". leak_plain is that protocol for the meters. One descriptive arm puts selection on it:
`analysis/selection_extraction.py --risky-model output/memorizing_llama8b --split attack_train --limit 100
--seed-tokens 20 --max-new-tokens 296 --n-values 1 8 64 --prefix selection_extraction_feat210 --out results`,
every other flag the script's default. Its passages and targets are built exactly as
`composition_attack.py` builds them (`join(prefix, reference)`, a `20`-token seed, the rest as target), and `296`
is the length leak_plain decoded every passage to. It reports the memoriser-ranked pick, the best of the `n`
draws (a selector that knows the answer) and the memoriser alone. No prediction is registered; it is
reported beside leak_plain whatever it reads. Host B, GPU 6.

### Scored 2026-09-25 (`analysis/score_feat210.py` -> `results/windowed_meter.csv`, `windowed_arms.csv`, `windowed_leakage.csv`)

**Gates.** G0 PASS: `14` generation arms cover the `500` prompts, `18` leakage runs the `100` passages.
G1 PASS: `0` of `7,000` trajectories and `0` leakage queries over their certified budget, each recomputed from
the per-step log (every windowed arm's worst span at most `W`: `4.11`, `12.48`, `24.95`, `40.0`, `124.71`
plain). G2 PASS: judge~B's plain pass reproduces the four committed arms' per-prompt levels on `500` of `500`
prompts, so it is the committed instrument.

**One deviation, stated before reading any comparison:** judge~G ran on host B (GPU 6), not the local A100,
because `google/gemma-2-27b-it` is not in the local cache. Every judge~G verdict of both passes comes from
that one host, so its differences are within one instrument; no judge~G level is set beside a committed one.

| | difference (plain, judge~B) | value [95%] | predicted | reading | verdict |
|---|---|---|---|---|---|
| P1 | selection - windowed `W=4.16` | `+0.090 [+0.058, +0.122]` | CONFIRMED | CONFIRMED | right |
| P1 | selection - front-loaded pathwise `4.16` | `+0.0785 [+0.0445, +0.1115]` | CONFIRMED | CONFIRMED | right |
| P2 | selection - windowed `W=24.95` | `+0.064 [+0.0315, +0.0975]` | CONFIRMED | CONFIRMED | right |
| P2 | selection - windowed `W=40` | `+0.064 [+0.0295, +0.099]` | CONFIRMED | CONFIRMED | right |
| P2 | selection - windowed `W=125` | `+0.022 [-0.0145, +0.0585]` | CONFIRMED | UNRESOLVED | **wrong** |
| P3 | installments `L=10` - pathwise `K=83.18` | `+0.142 [+0.1085, +0.1755]` | CONFIRMED | CONFIRMED | right |
| P3 | installments `L=25` - pathwise `K=33.27` | `+0.133 [+0.1015, +0.164]` | CONFIRMED | CONFIRMED | right |
| P4 | installments `L=10` - windowed `W=24.95` | `+0.1205 [+0.0855, +0.155]` | CONFIRMED | CONFIRMED | right |
| P4 | installments `L=25` - windowed `W=12.48` | `+0.1265 [+0.0915, +0.162]` | CONFIRMED | CONFIRMED | right |
| P5 | chat: selection - windowed `W=125` | `-0.160 [-0.1985, -0.122]` | REFUTED | REFUTED | right |
| P5 | chat: selection - windowed `W=12.48` | `+0.045 [+0.0095, +0.081]` | CONFIRMED | CONFIRMED | right |
| P6 | plain recall, `W<=40` / `W=125` | `0.000` / `0.106` | `<0.01` / `>=0.01` | | right |
| P6 | chat recall, `W<=40` / `W=125` | `0.000` / `0.063` | `<0.01` / `>=0.01` | | right |
| P7 | judge~G, P1-P4 same sign | `9` of `9` | all | | right |

**Thirteen of fourteen registered readings right; P2 at `W=125` wrong.** The windowed meter draws level with
selection at `125` nats a window, where it serves the risky model at `99.92%` of steps; we had predicted
selection would still lead, as it leads the `k=10` meter. Per the outcome rule fixed above, the paper states
that the windowed meter matches selection there, at that certificate (Section 4, `sec:matched`).

**The outcome rules, applied.** The matched comparisons are Section 4's first table (`tab:matched`; all
arms and both judges in `tab:matchedfull`), and Section 3's paragraph on designs the dichotomy leaves open
reports the windowed meter as measured. P5 holds but leak_chat reads `0.0632` at `W = 125`, not below
`0.01`, so the abstract and conclusion do **not** name the windowed meter as the mechanism for that regime:
the abstract says it wins there, certifies exact windows only, and leaks near-verbatim text at `125` nats,
where both judges prefer it. P6 holds in both configurations, so the paper states that an exact-window
certificate of `125` nats coexisted with near-verbatim extraction (`0.106` plain, `0.063` chat).

**Descriptive (no band).**

- Windowed onset, the smallest grid `W` with mean recall at least `0.01`: plain `80` (`0.0286`; `0.0063` at
  `60`), chat `125` (`0.0043` at `80`). The memoriser alone reads `0.4952` plain and `0.2193` chat; the
  anchor `0.0000`.
- Share of steps where the window binds (active) / serves the risky model unchanged, plain: `W=4.16`
  `100` / `0`; `12.48` `94.66` / `5.15`; `24.95` `68.77` / `31.18`; `40` `37.94` / `61.94`; `125` `0.08` /
  `99.92` (%). Chat `W=40`: `47.14` / `52.26`; `W=125`: `0.15` / `99.85`.
- Installments `L=10` against the KL meter at `k=0.5` (matched total nats, different orders):
  `+0.161 [+0.129, +0.1925]`.
- Headline (selection minus the `k=10` meter) on order-consistent prompts, judge~B: both compared arms
  consistent `+0.121 [+0.024, +0.222]` (`n = 62`); all four arms consistent `+0.0952 [0.000, +0.2143]`
  (`n = 21`). Judge~G, all prompts `+0.0545 [+0.0005, +0.108]`; consistent pair `+0.0276 [-0.0366, +0.0948]`
  (`n = 335`).
- Known quality difference, a second draw of the opponent's configuration minus the anchor: `+0.065
  [+0.0395, +0.090]` judge~B, `+0.1485 [+0.1045, +0.1915]` judge~G. Both judges detect it.
- The `k=0.5` null, `-0.003 [-0.0245, +0.0185]`: at `80%` power and two-sided `0.05` it could detect
  `0.0305` (`2.8016` standard errors); its `90%` interval `[-0.0205, +0.015]` lies inside both `+-0.065`
  (the known difference) and `+-0.0508` (half selection's gain), so it is equivalent to zero at both
  margins by two one-sided tests. Post hoc, a sensitivity reading and not observed power.
- Level of items whose served text is empty, plain: judge~B `0.42`-`0.50`, judge~G `0.62`-`0.75` (two arms
  with a single empty item read `1.0`); chat, both judges `0.19`-`0.29` (the chat anchor's two empties `0`).
  Judge~G scores an empty answer above the templateless opponent and far below the chat opponent.
- Chat, descriptive widths: selection minus windowed `W=24.95` `+0.0065 [-0.0305, +0.0435]` (B),
  `+0.053 [+0.011, +0.0945]` (G); `W=40` `-0.0525 [-0.0895, -0.015]` (B), `-0.0295 [-0.0725, +0.0145]` (G).
  So at `W = 40` through the chat template the meter wins under one judge and leaks nothing (`0.000`).

**The addendum's arm** (`results/selection_extraction_feat210.csv`, host B GPU 6): on leak_plain's own
passages and `296`-token targets, selection's near-verbatim recall is `0.0000` at `n = 1`, `8` and `64`, and
so is the best of its `n` draws (a selector that knows the answer). The memoriser alone reads `0.4729` here
against leak_plain's `0.4952`: two independent draws of the same configuration by two scripts with different
batching (caution (u)), not a disagreement about the corpus.

**Post hoc, labelled as such (not registered):** each registered difference re-read with every empty
served text counted as a loss (level `0`) instead of its judged level. Both chat readings (P5) stay resolved in
their registered direction under both judges. Of the plain ones judge~G still resolves every one but
`W = 125`; judge~B resolves only selection against the windowed meter at `log 64`
(`+0.048 [+0.0095, +0.0865]`) and installments against the per-token pathwise meters (`+0.0945`, `+0.094`).
The rows are in `windowed_meter.csv` under `post hoc`.
