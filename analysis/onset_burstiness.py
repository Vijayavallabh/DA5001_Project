"""Is the onset residue the burstiness Proposition 2 already names? (plan v5 / feat-083)

The onset ratio takes two values across the seven pairs and Limitations says the residue is not
resolvable with the anchors we hold. Proposition 2 says a budget publishes a DRIFT where safety is
a WORKLOAD MAXIMUM, and names the gap: k_crit(x) >= s(x) + delta/N, with equality only when the
surprisal accumulates at a constant rate. So k_crit/s(x) is a per-pair measure of how bursty the
protected work's surprisal process is under that anchor -- already computed, per passage, for every
pair, in results/budget_path_*.csv.

A finer tokenizer cuts the same text into more, individually less informative tokens, and a token
bucket that meters a mean rate is the object that handles burstiness badly. If that is the residue,
a burstier pair needs its budget raised further above s(x) before the work leaks, which is what an
onset ratio above 1 is. Predicted sign: POSITIVE.

Bands, and the statistic, committed in results/onset_prediction_burstiness.md BEFORE this ran.
Writes <out>/onset_burstiness.csv. No GPU.

  .venv/bin/python analysis/onset_burstiness.py --out results
"""
from __future__ import annotations

import argparse, csv, itertools, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset_gutenberg import exact_p, spearman  # noqa: E402

MANIFEST = "results/onset_pairs.tsv"


def load_pairs(path):
    """name<TAB>composition_summary.csv<TAB>budget_path.csv[<TAB>tokenizer<TAB>theory name]."""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        out.append((f[0], f[2]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--onset", default="results/onset.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    onset = {r["pair"]: r for r in csv.DictReader(open(a.onset)) if r["mode"] == "single"}
    rows = []
    for name, bp in load_pairs(a.manifest):
        if name not in onset or not os.path.exists(bp):
            print(f"[ob] skipping {name}", file=sys.stderr)
            continue
        d = list(csv.DictReader(open(bp)))
        s = st.median(float(r["s_mean"]) for r in d)
        kc = st.median(float(r["k_crit"]) for r in d)
        o = onset[name]
        rows.append(dict(
            pair=name, n=len(d),
            s_mean=round(s, 4), k_crit=round(kc, 4),
            burstiness=round(kc / s, 4),
            onset=round(float(o["onset_est"]), 4),
            ratio=round(float(o["onset_est_over_s"]), 4),
            onset_over_k_crit=round(float(o["onset_est"]) / kc, 4)))
    if len(rows) < 3:
        print("[ob] too few pairs", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: r["burstiness"])

    x = [r["burstiness"] for r in rows]
    y = [r["ratio"] for r in rows]
    rho, p = spearman(x, y), exact_p(x, y)
    cv = lambda v: st.stdev(v) / st.mean(v)
    cv_s = cv([r["ratio"] for r in rows])
    cv_k = cv([r["onset_over_k_crit"] for r in rows])

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "onset_burstiness.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("Proposition 2's own quantity against the onset ratio, over the pairs of the main table.")
    print("Bands committed in results/onset_prediction_burstiness.md before this ran.\n")
    print(f"{'pair':38s}{'s(x)':>8s}{'k_crit':>9s}{'k_crit/s':>10s}{'onset/s':>9s}{'onset/k_crit':>14s}")
    for r in rows:
        print(f"{r['pair'][:37]:38s}{r['s_mean']:>8.3f}{r['k_crit']:>9.3f}"
              f"{r['burstiness']:>10.3f}{r['ratio']:>9.3f}{r['onset_over_k_crit']:>14.3f}")
    verdict = ("NAMED: the residue is burstiness, and it is Proposition 2's own quantity"
               if rho >= 0.786 else
               "REFUTED: the residue stays unexplained" if abs(rho) < 0.6 else
               "INCONCLUSIVE at this many pairs; reported with its p and not built on")
    print(f"\nSpearman(k_crit/s, onset/s) over {len(rows)} pairs: rho = {rho:+.3f}, "
          f"exact p = {p:.4f}\n  -> {verdict}")
    print(f"\nsecondary: coefficient of variation across the pairs")
    print(f"  onset / s(x)      {cv_s:.4f}")
    print(f"  onset / k_crit    {cv_k:.4f}   -> k_crit is the "
          f"{'BETTER' if cv_k < cv_s else 'WORSE'} normaliser")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
