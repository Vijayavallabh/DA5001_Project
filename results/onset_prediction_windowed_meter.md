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
