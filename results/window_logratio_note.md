# What a windowed pathwise meter would spend — a POST-HOC measurement, no committed bands

Not a pre-registration. A referee (2026-09-24, sixth round) proposed a per-token design the paper's
dichotomy does not cover: cap each step's max-divergence by the budget left in a sliding `w`-token
window and deduct only the realised log-ratio `log p_r(y_t) - log p_s(y_t)`. Every window is then
certified at the window's budget; whether that budget binds on ordinary traffic is empirical.

`.venv/bin/python analysis/window_logratio.py --out results` -> `results/window_logratio.csv`. On the
risky model's own draws (`output/sweep_plain`, `k=-1`, the three ordinary classes, every seed; `1,500`
trajectories, `226,136` windows of `50` tokens, end-of-text padding dropped):

| quantity | median | p90 | p99 | max | share `>= S_w` |
|---|---|---|---|---|---|
| realised log-ratio per window | `40.3` | `63.9` | `92.5` | `188.7` | `0.011%` |
| per-step KL summed per window | `40.7` | `59.6` | `85.8` | `171.7` | `0.0097%` |
| largest window per trajectory | `63.2` | `87.9` | `125.3` | `188.7` | `0.067%` |

(`S_w = 159.83` nats, the audited anchor's median surprisal of a `50`-token window.)

**Reading.** A windowed pathwise meter with a `50`-token budget of about `40` nats would bind on half of
ordinary windows, and one that never binds needs about `125` nats, `0.78 S_w`. So it certifies exact
windows at a budget below their surprisal, which the dichotomy does not forbid because the budget is
per window rather than per sequence, but it cannot certify a sampled (near-verbatim) event: bounding one
at `1%` needs `N > 3 e^K` clean anchor draws, about `7 x 10^17` at `K = 40`. The paper states this in
Section 3 (`sec:uncovered`) and Appendix `app:windowlr`.
