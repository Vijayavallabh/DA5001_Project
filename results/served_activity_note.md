# Table 3's "active" column, on the trajectories the judge scored — a definition fix, no bands

Not a pre-registration. A referee (2026-09-24, sixth round) found Table 3's binding share (`5.5%` at
`k=1`) disagreeing with Appendix G's `beta` (`0.2057` at `k=1`). Both were right about different
things: Table 3's column came from `analysis/budget_calibration.py`, pooling the aggregate counters of
every class of `output/sweep_plain`, protected passages included (caution (l): those counters are
unreliable in that run), while `beta` counts every step the bucket constrains, strict blends plus tokens
the anchor writes outright, averaged per trajectory.

`.venv/bin/python analysis/served_activity.py --out results` -> `results/served_activity.csv` fixes one
definition and one population: **active** = decode steps whose served token comes from a strict blend of
the two models (`0 < bd < 1`), over all decode steps with end-of-text padding removed
(`dap.stats.strip_pad_steps`), pooled over the lowest-seed trajectory of each of the `500` ordinary
prompts, which is exactly what `analysis/order_averaged_h2h.py` serves to the judge.

| block | `k` | active | forced to the anchor |
|---|---|---|---|
| 8B-Instruct, continuing text | `0.5` / `1` / `10` | `49.1%` / `6.6%` / `0.008%` | `3.4%` / `1.5%` / `0.0%` |
| 70B base | `0.5` / `1` / `20` | `26.5%` / `6.3%` / `0.00%` | `4.6%` / `2.2%` / `0.0012%` |
| 8B-Instruct, chat template | `1` / `10` | `16.8%` / `0.16%` | `4.6%` / `0.31%` |

Table 3 prints this column with the definition in its caption; Appendix G's `beta` is unchanged and its
caption says what it counts that this column does not.
