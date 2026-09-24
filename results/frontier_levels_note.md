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
