# Adaptive (early-stopped) selection, reward only

Derived analysis over the committed reward cache `results/selection_rewards64.csv` (TinyComma
anchor, 500 prompts x 64 draws, pointwise Qwen2.5-7B reward). No generation, **no judge**. Written
and committed before the script was run; nothing above `## Scoring log` is edited afterwards.
Descriptive, so a `_note.md` and not an `onset_prediction_*.md`.

Produced by `.venv/bin/python analysis/adaptive_n.py --out results` -> `results/adaptive_n.csv`.

## Why

Proposition 1's hypothesis is only *serve one of the draws*, so a rule that draws one at a time and
stops as soon as a draw scores `>= tau`, capped at `n_max`, is certified at `log n_max` however
early it stops. If typical prompts stop early, the compute price (Section 3: the draws are the
price) falls while the certificate is unchanged.

## Rule

Draws are taken in the cache's `rank` order `0, 1, ...`. Stop at the first rank whose reward is
`>= tau` and serve it; if none of the first `n_max` qualifies, serve the argmax of those `n_max`
(first index on ties). `n_max in {8, 64}`. `tau` is a global quantile of the **rank-0** reward
distribution, `q in {0.50, 0.75, 0.90, 0.95, 0.99}`, fixed before reading any served result.

## Reported per (`n_max`, `q`)

* mean and median draws used, and the fraction stopping on the first draw;
* mean served reward, and the paired difference against (a) fixed-`n_max` argmax and (b) fixed-`n`
  argmax at `n = round(mean draws)` (the compute-matched fixed rule), bootstrapped over prompts
  (10,000 resamples, seed 0, 95% percentile);
* empty-served fraction (`n_words == 0`);
* the certificate, `log n_max` (pathwise), unchanged by `tau`.

## Caveats fixed in advance

* Reward, not utility. The reward is the selector's own score, so a gain in it is not a judged
  gain and it overoptimises on some tasks (TriviaQA, CoTaEval news).
* **The judged pass, fixed now, before the reward-only table exists:** two rules, `n_max = 64` at
  `q = 0.75` and `q = 0.90`, each with its compute-matched fixed-`n` arm (`n = round(mean draws)`)
  and fixed `n = 64`, judged by `analysis/levels_pass.py` (judge B, the committed opponent, both
  orders, de-echoed) on the first free card. Contrasts: adaptive minus fixed-`64` (equal certificate)
  and adaptive minus its compute-matched fixed `n` (equal expected draws), paired over prompts.
* Rewards are bf16 and reproducible to about `0.09` nats (caution (as)); differences below that
  are not read.

## Scoring log

### Reward-only table, 2026-09-24 14:40 IST --- stopping early saves draws and buys nothing at matched compute

`analysis/adaptive_n.py --out results` -> `results/adaptive_n.csv`. Reward nats, paired over the
`500` prompts, 95% bootstrap.

- **Against fixed `n_max` (equal certificate) every rule loses reward**, by design: at `n_max = 64`,
  `q = 0.75` stops after `6.77` draws on average (median `3`, `25.2%` on the first) and serves
  `-9.80 [-10.49, -9.14]` below the fixed-`64` argmax; `q = 0.99` uses `45.5` draws and gives back
  `-1.05 [-1.32, -0.80]`.
- **Against fixed `n` at the same expected draws (equal compute) adaptivity helps only at the lowest
  budget.** `q = 0.50` (`2.5` draws) beats fixed `n=2` by `+1.08 [+0.56, +1.61]`; `q = 0.75` (`6.8`)
  is **below** fixed `n=7`, `-0.62 [-1.24, -0.01]`; `q = 0.90` (`17.4`) straddles against fixed `17`,
  `-0.49 [-1.07, +0.07]`; `q >= 0.95` straddles. A threshold on the reward throws away the argmax's
  information about the draws it never took, and at moderate `n` that costs more than it saves.
- Empty-served fraction `4.4%`--`7.2%` throughout; certificate `log n_max` in every row.

Picks for the judged pass written to `results/adaptive_picks_*.csv` (the two registered rules and
their matched fixed-`n` arms, `n = 7` and `n = 17`).
