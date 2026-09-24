# Note (feat-186): order-averaged levels for Figure 1(b), one opponent, de-echoed text

**Committed 2026-09-24, before any judge call. Descriptive: no band, nothing is read as a test**, so
this is a note and not an `onset_prediction_*.md` (caution (as)).

**Why.** Figure 1(b) plotted the metered decoder's judged levels from one single-order pass
(`results/judge_separation_v6_judge2.csv`) beside selection's from another
(`results/selection_scaling.csv`), on text carrying the prompt's tail. Two single-order passes on one
axis is what caution (ap) forbids, and a referee noticed the panel plots levels while the text says
levels are never quoted.

**What runs.** `analysis/frontier_levels.py`, judge B (`microsoft/Phi-3.5-mini-instruct`), every item
in both presentation orders, against the committed opponent (`output/sweep_plain` `k=-1`, lowest
seed), all text de-echoed (`dap.shared.served_generation`), the corpus prompt shown to the judge:

- selection at `n in {1, 2, 4, 8, 16, 32, 64}`, picks replayed from `results/selection_rewards64.csv`;
- the metered decoder at `k in {0.5, 1, 3, 5, 10, 20}` from `output/phase2/conc_all`, lowest seed;
- the anchor alone, `output/sweep_plain` `k=0`, lowest seed.

x is selection's closed-form KL bound `log n - (n-1)/n` and the meter's measured mean spend over
all `1,500` trajectories at that `k` (the published `171.3` at `k=10`), as before.

**Determinism check, read first.** `sel_n64`, `sel_n1`, `met_k10` and `anchor_k0` were judged in
feat-184 Part A under the same construction; their per-prompt levels here must equal
`results/order_averaged_h2h_per_prompt_deecho.csv` exactly. If they do not, the panel is not redrawn
from this pass.

**What the figure does with it.** Panel (b) plots these levels, which are comparable to one another
because they share one opponent, one judge and both orders; its caption says levels are comparable
within this panel and nowhere else. The frontier curve is recomputed from the anchor arm's
order-averaged per-prompt distribution in this pass.

## Result (2026-09-24 11:20 IST)

All three passes exited `0` (`output/logs/feat186_{a,b,c}.done`, `scripts/run_feat186.sh`); merged by
`.venv/bin/python analysis/frontier_levels.py --merge a,b,c --out results` -> `results/frontier_levels.csv`.

**Determinism check: PASS.** `sel_n64`, `sel_n1`, `met_k10` and `anchor_k0` equal feat-184 Part A's
per-prompt levels on `500/500` prompts each.

Levels against the risky model's own draw (`0.5` = parity), with the gain over the anchor-alone arm
and selection at `n=64` minus the arm, both paired over the `500` prompts:

| arm | x (nats) | level | gain over anchor | `sel_n64` minus it |
|---|---|---|---|---|
| anchor alone | 0 | 0.4385 | --- | +0.1165 [+0.094, +0.1385] |
| meter `k=0.5` | 78.97 spent | 0.4355 | -0.0030 [-0.0235, +0.018] | +0.1195 [+0.0955, +0.1435] |
| meter `k=1` | 134.57 | 0.4555 | +0.0170 [-0.0055, +0.0405] | +0.0995 |
| meter `k=3` | 165.00 | 0.4745 | +0.0360 [+0.011, +0.0625] | +0.0805 |
| meter `k=5` | 169.85 | 0.4875 | +0.0490 | +0.0675 |
| meter `k=10` | 171.28 | 0.4895 | +0.0510 [+0.0255, +0.077] | +0.0655 [+0.040, +0.092] |
| meter `k=20` | 171.30 | 0.4895 | +0.0510 | +0.0655 |
| selection `n=1` | 0 (bound) | 0.4535 | +0.0150 [-0.007, +0.0365] | +0.1015 |
| `n=2` | 0.193 | 0.4730 | +0.0345 | +0.0820 |
| `n=4` | 0.636 | 0.4870 | +0.0485 [+0.027, +0.0695] | +0.0680 |
| `n=8` | 1.204 | 0.5115 | +0.0730 | +0.0435 |
| `n=16` | 1.835 | 0.5320 | +0.0935 | +0.0230 |
| `n=32` | 2.497 | 0.5535 | +0.1150 | +0.0015 [-0.0125, +0.0155] |
| `n=64` | 3.175 | 0.5550 | +0.1165 [+0.094, +0.139] | --- |

Read descriptively: the meter's gain over its own anchor is indistinguishable from zero at `k <= 1`
and plateaus at `+0.051` once it is the risky model (`k >= 5`); selection passes that plateau by
`n = 4` for a KL bound of `0.64` nats and passes parity with the risky model at `n = 8`. Selection at
`n = 64` is above the meter at every `k` on the grid. The four-arm pass's registered `D3` (`+0.0505`,
gains over each arm's OWN control) differs from `+0.0655` here only by the two anchor-alone controls
(`sel_n1` against `anchor_k0`, `0.4535` against `0.4385`), which is why this table uses one control.
