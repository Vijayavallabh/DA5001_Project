"""Plan v4 / feat-041: the published budget is not a sufficient statistic for the protection.

Proposition 1 says the vacuity threshold of the certificate is K = S(x) for every Renyi order
alpha in [1, inf]. Two decoders that publish the same k therefore publish the same certificate,
and at the budgets the mechanism's authors use that certificate is vacuous for both. This script
measures what those two decoders actually deliver.

It merges the per-order composition runs under output/phase4/renyi_*/ with the certificate status
from results/certificate_cap_summary.csv, so each row carries: the order, the budget, the measured
near-verbatim recall, and whether the certificate said anything at all. alpha = 1 is the audited
KL decoder and reproduces its published numbers, which is the cross-check that the independent
bisection solver is right.

Writes results/renyi_sweep.csv.
Usage: .venv/bin/python analysis/renyi_sweep.py --runs output/phase4 --out results
"""
import argparse, csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def order_of(dirname):
    """output/phase4/renyi_2 -> 2.0 ; renyi_1_0 -> 1.0 ; pathwise -> inf"""
    tag = os.path.basename(dirname)
    if "pathwise" in tag:
        return float("inf")
    m = re.search(r"renyi_(\d+)(?:_(\d+))?$", tag)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2) or '0'}")


def vacuous_pct(cap_summary, k):
    for r in cap_summary:
        if r.get("split") == "all" and abs(float(r["k"]) - k) < 1e-9:
            return float(r["vacuous_pct"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="output/phase4")
    ap.add_argument("--out", default="results")
    ap.add_argument("--caps", default="results/certificate_cap_summary.csv")
    a = ap.parse_args()

    caps = list(csv.DictReader(open(a.caps))) if os.path.exists(a.caps) else []
    rows = []
    for d in sorted(glob.glob(os.path.join(a.runs, "renyi_*"))):
        alpha = order_of(d)
        path = os.path.join(d, "composition_summary.csv")
        if alpha is None or not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            k = float(r["k"])
            if k <= 0:                                  # k=-1 and k=0 are the unbudgeted baselines
                continue
            rows.append({
                "alpha": alpha, "k": k, "mode": r["mode"], "L": int(r["L"]),
                "n_passages": int(r["n_passages"]),
                "nv_recall_mean": float(r["nv_recall_mean"]),
                "lcs_word_mean": float(r["lcs_word_mean"]),
                "max_query_Z_over_K": float(r["max_query_Z_over_K"]),
                "invariant_violations": int(r["invariant_violations"]),
                "certificate_vacuous_pct": vacuous_pct(caps, k),
            })
    if not rows:
        print(f"no renyi runs found under {a.runs}", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: (r["k"], r["mode"], r["L"], r["alpha"]))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "renyi_sweep.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    orders = sorted({r["alpha"] for r in rows})
    print(f"{'k':>4s} {'mode':>7s} {'L':>3s} {'cert vacuous':>13s}  " +
          "  ".join(f"a={o:g}".rjust(8) for o in orders))
    seen = {}
    for r in rows:
        seen.setdefault((r["k"], r["mode"], r["L"]), {})[r["alpha"]] = r
    for key in sorted(seen):
        k, mode, L = key
        g = seen[key]
        vac = next(iter(g.values()))["certificate_vacuous_pct"]
        cells = "  ".join((f"{g[o]['nv_recall_mean']:.3f}" if o in g else "-").rjust(8) for o in orders)
        print(f"{k:4g} {mode:>7s} {L:3d} {('%.0f%%' % vac) if vac is not None else 'n/a':>13s}  {cells}")
    print(f"\nwrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
