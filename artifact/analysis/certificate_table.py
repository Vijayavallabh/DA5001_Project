"""What a certificate buys a rights-holder, per response: the numbers behind the main text's
certificate table. No GPU, no new measurement.

  metered decoder (anchored decoding, KL charge) at k in {0.5, 3, 10}, T_max = 200: K = k T_max,
    the tightest bound on Pr[output reproduces a 50-token window] is the binary-KL inversion
    max{p : d(p || e^-S_w) <= K} (Proposition 2, alpha = 1), and the measured mean spend comes from
    results/imitation_cost.csv (ordinary traffic).
  selection at n in {8, 64}: K = log n under the pathwise order, bound min(1, n e^-S_w).
S_w is the audited anchor's median surprisal of a 50-token window (results/window_vacuity.csv).
Responses under a per-user cap of 400 nats are floor(400 / charge), charged at the certificate and,
for the meter, at its measured spend.

Usage: .venv/bin/python analysis/certificate_table.py --out results
"""
import argparse
import csv
import math
import os

T_MAX, CAP = 200, 400.0


def kl_bound(K, S):
    """max{p : d(p || e^-S) <= K}, the tightest event bound a KL budget K implies."""
    if K >= S:
        return 1.0
    lq = -S                                            # log q0

    def d(p):
        return p * (math.log(p) - lq) + (1 - p) * math.log1p(-p) if p < 1 else S
    lo, hi = math.exp(lq), 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if d(mid) <= K else (lo, mid)
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    w = {r["event"]: r for r in csv.DictReader(open(os.path.join(a.out, "window_vacuity.csv")))}
    S = float(w["50-token window"]["S_median_nats"])
    spend = {float(r["k"]): float(r["spend_nats"]) for r in csv.DictReader(
        open(os.path.join(a.out, "imitation_cost.csv"))) if r["prompt_class"] == "ordinary"}
    rows = []
    for k in (0.5, 3.0, 10.0):
        K = k * T_MAX
        b = kl_bound(K, S)
        rows.append(dict(mechanism="metered", param=f"k={k:g}", order="KL", certificate_nats=round(K, 4),
                         K_over_Sw=round(K / S, 6), window_bound=round(b, 6),
                         window_bound_log10=round(math.log10(b), 4), measured_spend_nats=spend[k],
                         responses_at_certificate=math.floor(CAP / K),
                         responses_at_spend=math.floor(CAP / spend[k])))
    for n in (8, 64):
        K = math.log(n)
        lb = min(0.0, (K - S) / math.log(10))
        rows.append(dict(mechanism="selection", param=f"n={n}", order="pathwise",
                         certificate_nats=round(K, 4), K_over_Sw=round(K / S, 6),
                         window_bound=10 ** lb, window_bound_log10=round(lb, 4),
                         measured_spend_nats="", responses_at_certificate=math.floor(CAP / K),
                         responses_at_spend=""))
    path = os.path.join(a.out, "certificate_table.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(r)
    print(f"S_w = {S}; wrote {path}")


if __name__ == "__main__":
    assert kl_bound(200.0, 100.0) == 1.0 and abs(kl_bound(50.0, 100.0) - 0.5) < 0.02
    main()
