"""feat-111: what the log n certificate delivers when its premise fails.

Proposition 4 bounds an EVENT: Pr_q[E] <= n Pr_{p_s}[E] for every E. Every leakage arm in this
paper has a clean anchor, so Pr_{p_s}[E] = 0 and the inequality is satisfied by any number we
could have measured. These arms use anchors that have seen the work, so the base rate is non-zero
and the bound is testable for the first time.

Reads the per-passage CSVs written by analysis/selection_extraction.py (one per contaminated
anchor) and writes <out>/contaminated_anchor.csv. Bands and their readings are
results/onset_prediction_contaminated_anchor.md; this script computes them, it does not choose
them.

Usage:
  .venv/bin/python analysis/contaminated_anchor.py --out results
"""
import argparse
import csv
import glob
import math
import os
import re

# The events Proposition 4 is stated for. A mean recall is not an event probability and is
# reported beside these, never in place of them.
EVENTS = (("E_08", 0.8), ("E_001", 0.01))
N_GRID = (1, 8, 64)
SATURATION_BAND = 4.0          # N2: A(64) <= 4 reads SATURATES


def rate(recalls, thresh):
    """Pr[nv_recall >= thresh] over the passages, and its binomial standard error."""
    n = len(recalls)
    p = sum(1 for r in recalls if r >= thresh) / n
    return p, math.sqrt(p * (1.0 - p) / n)


def spearman(xs, ys):
    def rank(z):
        order = sorted(range(len(z)), key=lambda i: z[i])
        out = [0.0] * len(z)
        for j, i in enumerate(order):
            out[i] = float(j)
        return out
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def arms(results_dir):
    """(tag, rows) per contaminated anchor, ordered by tag so the CSV is stable."""
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "contam_*_per_passage.csv"))):
        tag = re.sub(r"^contam_|_per_passage\.csv$", "", os.path.basename(path))
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, path
        out.append((tag, rows))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    found = arms(a.results)
    assert found, f"no contam_*_per_passage.csv in {a.results}"

    out, violations, amps = [], [], {}
    for tag, rows in found:
        recalls = {n: [float(r[f"recall_n{n}"]) for r in rows] for n in N_GRID}
        risky = [float(r["risky_alone_recall"]) for r in rows]
        for name, thresh in EVENTS:
            base, base_se = rate(recalls[1], thresh)
            for n in N_GRID:
                p, se = rate(recalls[n], thresh)
                bound = min(1.0, n * base)
                # N1 is read with the binomial slack of the MEASURED rate, as registered.
                ok = p <= bound + se
                if not ok:
                    violations.append((tag, name, n, p, bound, se))
                out.append(dict(
                    anchor=tag, event=name, threshold=thresh, n=n, n_passages=len(rows),
                    rate=round(p, 4), rate_se=round(se, 4),
                    base_rate=round(base, 4), base_rate_se=round(base_se, 4),
                    bound_n_times_base=round(bound, 4),
                    amplification=(round(p / base, 4) if base > 0 else ""),
                    prop4_holds=("yes" if ok else "NO"),
                    mean_recall=round(sum(recalls[n]) / len(rows), 4),
                    risky_alone_mean_recall=round(sum(risky) / len(risky), 4)))
                if name == "E_08" and n == 64 and base > 0:
                    amps[tag] = (base, p / base)

    path = os.path.join(a.out, "contaminated_anchor.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    for r in out:
        if r["n"] == 1:
            continue
        print(f"  {r['anchor']:12s} {r['event']:6s} n={r['n']:>3}  rate={r['rate']:.3f} "
              f"[base {r['base_rate']:.3f}]  bound {r['bound_n_times_base']:.3f}  "
              f"A={r['amplification'] or '--':>6}  Prop4 {r['prop4_holds']}")
    print()
    print(f"  N1 {'HOLDS' if not violations else 'VIOLATED'}"
          + ("" if not violations else f" at {violations}"))
    if amps:
        worst = max(amps.items(), key=lambda kv: kv[1][1])
        reading = "SATURATES" if worst[1][1] <= SATURATION_BAND else "GROWS"
        print(f"  N2 {reading}: largest A(64) on E_08 is {worst[1][1]:.2f} at {worst[0]} "
              f"(band {SATURATION_BAND}, permitted 64)")
        if len(amps) >= 3:
            xs = [v[0] for v in amps.values()]
            ys = [v[1] for v in amps.values()]
            print(f"  N3 descriptive, no band: Spearman(base rate, A(64)) = "
                  f"{spearman(xs, ys):+.3f} over {len(amps)} anchors")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
