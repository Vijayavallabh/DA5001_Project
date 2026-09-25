"""Score feat-209 (results/onset_prediction_nonempty_tables.md): each Table 2 row under the committed rule
and under the non-empty rule, judged in one pass on one host. No GPU. Writes results/nonempty_tables.csv.

Usage: .venv/bin/python analysis/nonempty_tables.py --out results
"""
import argparse
import csv
import os
import random

ROWS = ("headline", "he70b_k20", "he70b_k05", "chat_k10", "chat_k3", "t07_8b_k10", "t07_70b_k20", "ab70_k01")
N_BOOT = 10000


def label(lo, hi, v):
    return "UNRESOLVED" if lo <= 0 <= hi else "CONFIRMED" if v > 0 else "REFUTED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(209)
    out = []
    for row in ROWS:
        tag = f"nonempty_{row}"
        summ = {r["quantity"][:2]: r for r in csv.DictReader(open(os.path.join(a.results, f"order_averaged_h2h_{tag}.csv")))}
        per = list(csv.DictReader(open(os.path.join(a.results, f"order_averaged_h2h_per_prompt_{tag}.csv"))))
        met = next(c for c in per[0] if c.startswith("u_metered_k"))
        d = [(float(r["u_sel_nonempty"]) - float(r["u_sel_n1"])) - (float(r[met]) - float(r["u_anchor_k0"])) for r in per]
        n = len(d)
        xs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
        lo, hi, v = xs[int(0.025 * N_BOOT)], xs[int(0.975 * N_BOOT)], sum(d) / n
        c = summ["D3"]
        changed = sum(r["u_sel_nonempty"] != r["u_sel_n64"] for r in per)
        out.append(dict(row=row, n=n, D3=float(c["value"]), D3_lo95=float(c["lo95"]), D3_hi95=float(c["hi95"]),
                        D3_reading=label(float(c["lo95"]), float(c["hi95"]), float(c["value"])),
                        D3_nonempty=round(v, 4), D3_nonempty_lo95=round(lo, 4), D3_nonempty_hi95=round(hi, 4),
                        D3_nonempty_reading=label(lo, hi, v), prompts_level_moved=changed))
    for r in out:
        r["same_label"] = r["D3_reading"] == r["D3_nonempty_reading"]
    path = os.path.join(a.out, "nonempty_tables.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)
    print(f"{sum(r['same_label'] for r in out)} of {len(out)} rows keep their label; wrote {path}")


if __name__ == "__main__":
    main()
