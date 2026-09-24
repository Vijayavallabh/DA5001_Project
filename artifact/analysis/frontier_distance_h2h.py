"""How far each mechanism sits from Theorem 1's frontier on the headline pass (a referee question).

Appendix A prices the frontier at n=8 from one single-order pass (selection 58x, the meter's best arm
7,995x). A report asked the same of the headline, n=64: is log 64 near Lambda*_s or far from it?
This reads the order-averaged per-prompt utilities of that pass, takes each arm's OWN control as the
safe law of U (a five-point law, since averaging two orders puts U in {0, 1/4, 1/2, 3/4, 1}), and
divides each arm's divergence -- selection's KL bound, the meter's measured spend -- by
Lambda*_s of the arm's mean utility. Same pass, same judge, same instrument for both.

The rate function is utility_price.rate generalised from three points to any finite law; the
self-check below reproduces utility_price.rate on its own three-point law before anything is read.

No GPU.  Usage:  .venv/bin/python analysis/frontier_distance_h2h.py --out results
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.utility_price import rate as rate3  # noqa: E402

ARMS = (("selection, n=64", "u_sel_n1", "u_sel_n64", "KL bound", math.log(64) - 63 / 64),
        ("metered decoder, k=10", "u_anchor_k0", "u_metered_k10", "measured spend", 171.28))


def rate(law, u, lo=-200.0, hi=200.0, iters=300):
    """Lambda*(u) = sup_lam [lam u - log E e^{lam U}] for a finite law [(value, prob)]."""
    def f(lam):
        return lam * u - math.log(sum(p * math.exp(lam * v) for v, p in law))
    for _ in range(iters):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if f(m1) < f(m2):
            lo = m1
        else:
            hi = m2
    return max(0.0, f((lo + hi) / 2.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-prompt", default="results/order_averaged_h2h_per_prompt.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    w, t, l = 0.264, 0.110, 0.626                   # a three-point law, as utility_price uses
    assert abs(rate([(1.0, w), (0.5, t), (0.0, l)], 0.4) - rate3((w, t, l), 0.4)) < 1e-9

    rows = list(csv.DictReader(open(a.per_prompt, encoding="utf-8")))
    out = []
    for name, ctrl, arm, kind, nats in ARMS:
        n = len(rows)
        law = sorted((v, c / n) for v, c in Counter(float(r[ctrl]) for r in rows).items())
        u0 = sum(v * p for v, p in law)
        u = sum(float(r[arm]) for r in rows) / n
        lam = rate(law, u)
        out.append(dict(arm=name, control=ctrl, n_prompts=n, u_control=round(u0, 6),
                        u_arm=round(u, 6), gain=round(u - u0, 6), divergence_kind=kind,
                        divergence_nats=round(nats, 4), lambda_star_nats=round(lam, 6),
                        over_frontier=round(nats / lam, 1)))
        print(f"[frontier] {name:22s} gain {u - u0:+.4f}  Lambda* {lam:.5f} nats  "
              f"{kind} {nats:.4f}  => {nats / lam:,.1f}x the frontier")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "frontier_distance_h2h.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"[frontier] wrote {path}")


if __name__ == "__main__":
    main()
