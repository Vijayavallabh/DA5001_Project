"""Turn a pair's s(x) into the committed k-grid (plan v5 / feat-084).

The grid for a new pair has to be fixed before the sweep and cannot be written down in nats until
s(x) is measured, so results/onset_prediction_granularity_gap.md commits it as a RULE in units of
k/s(x) and this applies the rule. Choosing a grid after seeing where the curve crosses is the
failure the pre-registrations exist to prevent; doing the arithmetic by hand is how that failure
gets in by accident.

  .venv/bin/python analysis/grid_from_sx.py --budget-path results/budget_path_<tag>.csv
"""
from __future__ import annotations

import argparse, csv, statistics as st

# committed in results/onset_prediction_granularity_gap.md before any weights were downloaded
RULE = (0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.15, 1.30, 1.55)


def grid(s_x, rule=RULE):
    """The budgets in nats per token, rounded to two decimals, with both mandatory baselines."""
    return [-1.0, 0.0] + [round(f * s_x, 2) for f in rule]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-path", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.budget_path)))
    s_x = st.median(float(r["s_mean"]) for r in rows)
    k_crit = st.median(float(r["k_crit"]) for r in rows)
    g = grid(s_x)
    print(f"{len(rows)} passages: s(x) = {s_x:.4f} nats/token, k_crit = {k_crit:.4f} "
          f"(k_crit/s = {k_crit / s_x:.3f})")
    print(f"k/s(x) rule: {' '.join(f'{f:g}' for f in RULE)}")
    print(f"--k-values {' '.join(f'{k:g}' for k in g)}")


if __name__ == "__main__":
    main()
