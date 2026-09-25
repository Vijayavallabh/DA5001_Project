"""Figure 3's rows under a family-wise correction (Holm), for Review 3's W12/Q9. No GPU. Post hoc.

Each row of the head-to-head forest plot is a paired mean over prompts with a 95% bootstrap interval.
The p-value of a row is read off its own interval with the normal approximation the interval implies
(SE = (hi - lo) / (2 * 1.96)), two-sided; Holm's step-down procedure at alpha = 0.05 then runs over all
the figure's rows as one family. The rows are figures/make_figures_v4.h2h_forest_rows(), the function
that draws the figure, so the family is exactly what the reader sees.

Usage: .venv/bin/python analysis/forest_multiplicity.py --out results
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))


def p_from_interval(v, lo, hi):
    se = (hi - lo) / (2 * 1.959964)
    z = abs(v) / se if se > 0 else float("inf")
    return math.erfc(z / math.sqrt(2))


def holm(ps, alpha=0.05):
    """Reject flags in input order: step down over the sorted p-values, stop at the first acceptance."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, m = [False] * len(ps), len(ps)
    for j, i in enumerate(order):
        if ps[i] <= alpha / (m - j):
            rej[i] = True
        else:
            break
    return rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    from make_figures_v4 import h2h_forest_rows
    rows = h2h_forest_rows()
    ps = [p_from_interval(*r[2]) for r in rows]
    rej = holm(ps)
    out = []
    for (g, lab, (v, lo, hi)), p, r in zip(rows, ps, rej):
        out.append(dict(group=g, row=lab, value=v, lo95=lo, hi95=hi, excludes_zero_95=not (lo <= 0 <= hi),
                        p_normal=f"{p:.3g}", holm_reject=r, sign="+" if v > 0 else "-"))
    path = os.path.join(a.out, "forest_multiplicity.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    n95 = sum(r["excludes_zero_95"] for r in out)
    print(f"{len(out)} rows; {n95} exclude zero at 95%; {sum(rej)} survive Holm at 0.05")
    for r in out:
        print(f"  {r['row']:42s} {r['value']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}] p={r['p_normal']:>9s} "
              f"{'HOLM' if r['holm_reject'] else '    '}")
    print("wrote", path)


if __name__ == "__main__":
    main()
