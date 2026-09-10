"""Consolidate the greedy-vs-equal-price probes into one table (plan v5 / feat-070).

Globs results/mp_<tag>_k<k>.csv, written by analysis/marginal_price.py, and reports for each
(pair, target type, budget) how much fidelity an oracle reallocation of greedy's own total spend
would buy. Writes results/marginal_price_table.csv. No GPU.

  .venv/bin/python analysis/marginal_price_table.py --out results
"""
from __future__ import annotations

import argparse, csv, glob, os, re, statistics as st

# Three target types, and the distinction between the first two is not cosmetic. `_prot` runs used
# the `test` split, which every phase-5 memoriser holds out and whose novels it has never seen; `_mem`
# runs use `attack_train`, the split it was fine-tuned on and the trajectory an extraction adversary
# actually walks. See results/onset_prediction_orders_matched.md.
LABEL = {"kl3m_prot": ("KL3M-520M", "held-out novel"),
         "kl3m_mem": ("KL3M-520M", "memorised passage"),
         "kl3m_util": ("KL3M-520M", "ordinary generation"),
         "pleias_prot": ("Pleias-1.2B", "held-out novel"),
         "pleias_mem": ("Pleias-1.2B", "memorised passage"),
         "pleias_util": ("Pleias-1.2B", "ordinary generation")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.out, "mp_*_k*.csv"))):
        m = re.match(r"mp_(.+)_k([0-9.]+)\.csv$", os.path.basename(path))
        if not m:
            continue
        tag, k = m.group(1), float(m.group(2))
        pair, target = LABEL.get(tag, (tag, ""))
        r = list(csv.DictReader(open(path)))
        if not r:
            continue
        g = lambda key: [float(x[key]) for x in r]
        rows.append(dict(
            pair=pair, target=target, k=k, n=len(r),
            gain_ratio=round(st.mean(g("gain_ratio")), 4),
            gain_lo=round(min(g("gain_ratio")), 4), gain_hi=round(max(g("gain_ratio")), 4),
            frac_ceiling_greedy=round(st.mean(g("frac_of_ceiling_greedy")), 4),
            frac_ceiling_dual=round(st.mean(g("frac_of_ceiling_dual")), 4),
            theta_at_one_pct=round(st.mean(g("theta_greedy_at_one_pct")), 1),
            spend=round(st.mean(g("spend_greedy")), 2),
        ))
    rows.sort(key=lambda x: (x["pair"], x["target"], x["k"]))
    path = os.path.join(a.out, "marginal_price_table.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("What an oracle reallocation of greedy's own total spend buys, at a fixed trajectory.\n")
    print(f"{'pair':13s}{'target':22s}{'k':>6s}{'greedy % of ceiling':>21s}"
          f"{'oracle %':>10s}{'gain':>8s}{'[min,max]':>17s}{'theta=1':>9s}")
    for r in rows:
        print(f"{r['pair']:13s}{r['target']:22s}{r['k']:>6.2f}"
              f"{100*r['frac_ceiling_greedy']:>20.1f}%{100*r['frac_ceiling_dual']:>9.1f}%"
              f"{r['gain_ratio']:>8.3f}  [{r['gain_lo']:.3f},{r['gain_hi']:.3f}]"
              f"{r['theta_at_one_pct']:>8.1f}%")
    binding = [r["gain_ratio"] for r in rows if r["k"] <= 1.0]
    print(f"\nover the budgets where the certificate says anything (k <= 1): "
          f"gain {min(binding):.3f} to {max(binding):.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
