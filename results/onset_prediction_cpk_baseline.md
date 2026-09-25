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
