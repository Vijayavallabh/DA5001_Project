# Pre-registration: CP-k's once-per-output rejection rule as a baseline, run with the anchor as its safe model (feat-211)

**feat-211.** Committed **2026-09-25**, after a smoke of every code path on six prompts and two passages
(`output/feat211_smoke`, host B GPU 4) and before any token of the arms below. Nothing above `## Scoring log`
is edited after the first token.

## Why

Review 1 (Q1 and its roadmap) and review 3 (W1): CP-k (Vyas et al. 2023) already pays its budget once per
complete output, and the paper computed only the certificate it would carry, never ran it: "report acceptance
rate, certified K, leakage, and the same judge protocol on the 500 prompts". CP-k's premise, safe models trained
on disjoint shards, is met by no model pair here, so this runs CP-k's rejection rule with the one safe model the
paper has, the anchor, and says so.

## The mechanism, and its certificate

For a prompt `x` the risky model's draws `y_1, ..., y_M` (`M = 64`, in seed order) are tested in turn; the
first whose realised log-ratio `R(y) = log p_r(y|x) - log p_s(y|x)`, summed over the served tokens through the
terminating end-of-text, is at most `kappa` is served; if none passes, the anchor's own draw is served. For every
`y`,

    q(y) = sum_{j<=M} (1-a)^{j-1} p_r(y) 1[R(y) <= kappa] + (1-a)^M p_s(y) <= (M e^kappa + 1) p_s(y),

so `D_inf(q || p_s) <= C = log(1 + M e^kappa)` pathwise, for every prompt, whatever the acceptance rate `a`
(`tests/test_cpk_baseline.py` checks it exactly on small laws, and that it is approached as `a -> 0`). `kappa`
is set from `C`: `kappa = log((e^C - 1)/M)`.

Grid `C` (nats): `4.1589` (`log 64`, selection's), `33.271` (installments `L=25` over the output), `83.178`
(installments `L=10` over the output), `159.83` (the median protected window's `S_w`, where the certificate
says nothing about a window).

## What runs (host B, GPUs 4-7; `scripts/run_feat211.sh`)

- **Draws.** `h1.py --k-values -1 --trajectories-per-prompt 16 --seeds <b> --cap-neutral 200 --cap-factual 150
  --cap-creative 150 --cap-val 0 --cap-test 0 --cap-attack-train 0 --batch-size 48 --trust-remote-code`, one run
  per card with base seeds `1101`, `1102`, `1103`, `1104` (the harness derives trajectory seeds from the base, so
  the four sets are disjoint from each other and from every seed on record), the headline's `500` prompts,
  TinyComma-1.8B with Llama-3.1-8B-Instruct, temperature `1.0`, no penalty, `T_max = 200`, no chat template ->
  `output/feat211/risky_{a,b,c,d}`. Draw order is seed order.
- **Arms.** `analysis/cpk_baseline.py --risky-dirs output/feat211/risky_{a,b,c,d}` writes one arm per `C`
  (`output/feat211/cpk_arms`, token `cpk<C>`); the fallback is the committed anchor arm (`output/sweep_plain`,
  `k=0`, lowest seed), so each arm's zero-budget control is that same arm, paired.
- **Leakage.** `analysis/cpk_extraction.py --risky-model output/memorizing_llama8b` on the `100` passages of the
  leakage sweep (`attack_train`, `20`-token seeds, `296` new tokens, built by `selection_extraction.py`'s
  `build()`), `M = 64` memoriser draws each, `R` computed on the generated ids, grid `C` extended by `250`, `400`,
  `600`, `800`, fallback one anchor draw per passage.
- **Judging.** `analysis/matched_h2h.py`, both presentation orders, de-echoed text, `1,200` characters, the
  committed opponent. Two passes, both on host B: judge~B (`Phi-3.5-mini-instruct`) with a fresh verdict cache,
  tag `cpk_B_hostb` -- the committed judge-B pass ran on a local A100, and judged numbers are host-dependent
  (caution (as)), so every arm this pass compares is re-judged here and nothing in it is set beside a committed
  judge-B level; and judge~G (`gemma-2-27b-it`), tag `cpk_G`, whose cache is a copy of the committed
  `matched_plain_G` verdicts, judged on this host. Arms: `sel_n64`, `sel_n1`, `anchor_k0`, `metered_k10`,
  `opp_r1`, `blk10n64`, `blk25n64`, `blk200n1`, `win_0`, `win_4.16`, `frontpw_4.16`, `pw_83.18`, `pw_33.27`,
  the four `cpk_<C>`, and feat-212's `sel_g64`, each with its committed specification and control; `cpk_<C>`
  against `anchor_k0`.

## Gates

- **G0.** Every prompt has exactly `64` draws at distinct seeds; every passage `64` memoriser draws.
- **G1.** Every served risky or memoriser draw satisfies `R <= kappa` for its arm (the scripts assert it), and
  `log(1 + M e^kappa)` reproduces `C` to `1e-4`.

## Predictions (judge~B pass on host B unless named)

- **C1** matched whole-output certificate `83.178`: installments (`L=10`) minus CP-k: **CONFIRMED**.
- **C2** matched `33.271`: installments (`L=25`) minus CP-k: **CONFIRMED**.
- **C3** matched `log 64`: best-of-`64` minus CP-k: **CONFIRMED**.
- **C4** judge~G gives C1-C3 the **same sign**.
- **C5** CP-k pays the imitation cost per output, so it accepts short outputs: at `C = 83.178` the accepted
  draws' median true length (tokens) is **below** the median of all `32,000` draws.
- **C6** leakage: CP-k's near-verbatim recall is **below `0.01`** at every `C <= 159.83`.

**Descriptive, no band.** The share of prompts served from the risky model at each `C`; each arm's gain; CP-k
minus the per-token pathwise meter at `83.18` and `33.27` (same order, same nats, charged once against per
token); CP-k at `159.83` minus the `k=10` meter; the leakage onset (the smallest grid `C` whose recall reaches
`0.01`) and the memoriser alone; the share of the four committed arms' per-prompt levels this host-B judge-B
pass reproduces from the local one; empties served.

## What the manuscript does with each outcome, fixed now

Section 5's CP-k sentence reports the rule as run (certificate, share served from the risky model, gain, the
matched differences), in place of the certificate-only arithmetic, and an appendix paragraph and table carry
every row. Wherever C1-C3 hold, the paper says selection and installments beat CP-k's rule at a matched
certificate; wherever one fails, it states the certificate at which CP-k matches or beats them. If C5 holds,
the paper states that CP-k accepts only outputs short enough to pay the imitation cost once.

## Excluded in advance

- Any other `M`, grid point, seed, temperature, template, judge or fallback chosen after a token is decoded or a
  verdict read; pooling this pass with any other; setting a host-B judge-B level beside a committed one.

## Scoring log

### Scored 2026-09-25 (host B, judge~B and judge~G passes) --- five of six registered readings right

Command: `.venv/bin/python analysis/score_feat211.py` -> `results/cpk_baseline_scoring.csv` (every verdict computed
from `results/matched_h2h_cpk_{B_hostb,G}.csv`, `results/cpk_baseline.csv` and `results/cpk_extraction.csv`).

**A defect in our code, found and repaired before any verdict was read.** The first `arms` step summed `R`
only through the first `<|eot_id|>` (`analysis/window_logratio.trajectory_steps` treats `128009` as an end).
The plain harness stops only at `<|end_of_text|>`, so the model writes on after an `<|eot_id|>` and the served
text keeps those tokens. The flawed arm served the risky model on three prompts at `C = 33.27`, and all three
were such truncations: an eot_id at step `8` or `18`, then `132` to `154` words served. G1 held there on an
under-counted `R`, which is not what this document registers ("summed over the served tokens through the
terminating end-of-text"). Both judge passes were stopped while they were still judging non-CP-k arms. The
fix is `analysis/cpk_baseline.py:served_steps`, which sums through the harness's own end; a test fails on the
old rule. The arms were rebuilt and the passes re-entered (`scripts/run_feat211.sh q4b`/`q5b`); the non-CP-k
verdicts stayed cached. No CP-k verdict of the flawed arms was computed. The same reader cuts `3` of the
`1,500` committed `sweep_plain` `k=-1` trajectories the same way (`0.2%`), which is logged in `progress.md`
rather than changed here.

**Gates.** G0: every prompt has `64` draws at `64` distinct seeds, and every passage `64` memoriser draws
(the scripts assert both). G1: every served draw has `R <= kappa` (asserted), and `log(1 + M e^kappa)`
reproduces every grid `C` to `1e-4` (asserted by the scorer).

| | reading | registered | verdict |
|---|---|---|---|
| C1 installments `L=10` minus CP-k at `83.18` (B) | `+0.1205` `[+0.091, +0.1485]` CONFIRMED | CONFIRMED | RIGHT |
| C2 installments `L=25` minus CP-k at `33.27` (B) | `+0.143` `[+0.119, +0.168]` CONFIRMED | CONFIRMED | RIGHT |
| C3 best-of-`64` minus CP-k at `log 64` (B) | `+0.110` `[+0.0855, +0.135]` CONFIRMED | CONFIRMED | RIGHT |
| C4 the same three under G | `+0.254`, `+0.2855`, `+0.176`, all CONFIRMED | same sign | RIGHT |
| C5 accepted draws shorter at `83.18` | median `200` tokens, as all draws (`200`) | below | **WRONG** |
| C6 leakage below `0.01` at every `C <= 159.83` | `0.0000` at all four | `< 0.01` | RIGHT |

**Descriptive.**
- Share of prompts CP-k serves from the risky model: `0.0%` at `log 64` and at `33.27`, `26.8%` at
  `83.18` and `99.6%` at `159.83`. Where it serves the anchor on every prompt, the arm is the anchor arm
  itself (gain `0.000`).
- Gains over the anchor: `+0.0395` `[+0.0245, +0.0555]` (B) and `+0.051` (G) at `83.18`; `+0.0865`
  `[+0.062, +0.1105]` and `+0.1335` at `159.83`.
- CP-k minus the per-token pathwise meter at the same nats: `+0.022` `[-0.006, +0.0495]` at `83.18` and
  `-0.014` `[-0.0335, +0.006]` at `33.27`, both unresolved (G: `+0.0385`, `+0.0185`).
- CP-k at `159.83` minus the `k=10` meter: `+0.0355` `[+0.009, +0.062]` under B and `+0.012`
  `[-0.0335, +0.057]` under G.
- Leakage: the memoriser alone reads `0.4629` on these draws. CP-k's recall is `0.0000` through `C = 250`,
  `0.0016` at `400` (`3%` of passages served from the memoriser), `0.0563` at `600` (`80%`) and `0.2002` at
  `800` (`100%`), so its onset on this grid is `600` nats. The median `R` of a memoriser draw here is in
  `cpk_extraction.csv`, and the median `R` of a risky draw on the judged workload is `169.6` nats.
- This host-B judge-B pass gives the local pass's per-prompt level on `93.4%` to `94.8%` of prompts for
  the four committed arms.

**What the manuscript does (as fixed above).** Section 5's CP-k sentence reports the rule as run: it
serves the risky model on no prompt at `log 64` or `33` nats and on `26.8%` at `83`, and at every matched
certificate best-of-`64` and installments beat it (C1-C4). The paper states that C5 failed: CP-k does not
buy its acceptances with short outputs here, because nearly every draw runs to the `200`-token cap.
