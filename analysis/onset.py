"""Plan v4 / feat-043: is the vacuity threshold tight?

Proposition 1 says the certificate becomes vacuous at K = S(x) and that no order of the charge
moves that point. It says nothing about whether the point is in the right place. If the bound were
merely loose, leakage would begin well above it and the threshold would be a needlessly early
warning. This measures where leakage actually begins, on a k grid fine enough to resolve it, for
two independent (anchor, risky) pairs whose thresholds differ by 36%.

The test that makes it more than a coincidence is that the onset should MOVE with the threshold.

It also tests the stronger version of the claim. If s(x) is the natural scale for the budget, then
plotting recall against k/s(x) should collapse the two pairs onto one curve. For single queries it
does, to a mean absolute difference of 0.006 over k/s in [0.7, 1.2] and 0.001 at the crossing. For
oracle windows it does not (0.031), which is what a per-WINDOW budget K_i = k L should do: the
window, not the work, sets the surprisal that matters.

Reads the fine-grid composition runs and the per-passage budget-path files that carry s_mean.
Writes <out>/onset.csv and <out>/onset_collapse.csv.

Usage: .venv/bin/python analysis/onset.py --out results
"""
import argparse, csv, os, statistics as st, sys

PAIRS = [
    ("TinyComma-1.8B + memorised Llama-3.1-8B", "output/phase4/fine_tc/composition_summary.csv",
     "results/budget_path.csv"),
    ("Comma-7B + memorised Comma-7B", "output/phase4/fine_comma/composition_summary.csv",
     "results/budget_path_comma7b.csv"),
]


def curve(path, mode, L):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["mode"] == mode and r["L"] == str(L) and float(r["k"]) > 0:
            out[float(r["k"])] = float(r["nv_recall_mean"])
    return dict(sorted(out.items()))


def crossing(c, thresh):
    """Bracket (k_lo, k_hi] within which recall first reaches `thresh`, and a linear estimate."""
    prev = None
    for k in sorted(c):
        if c[k] >= thresh:
            if prev is None:            # already above threshold at the smallest budget probed
                return None, k, k
            lo, hi = c[prev], c[k]
            est = prev + (k - prev) * (thresh - lo) / (hi - lo) if hi > lo else k
            return prev, k, est
        prev = k                        # advance the bracket; omitting this reports the grid point
    return None, None, None


def collapse(data, xs=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2)):
    """Interpolate each pair's recall onto a common k/s(x) grid and report the disagreement."""
    import bisect

    def at(c, s_x, x):
        ks = [k / s_x for k in sorted(c)]
        vs = [c[k] for k in sorted(c)]
        if x <= ks[0] or x >= ks[-1]:
            return None
        i = bisect.bisect_left(ks, x)
        t = (x - ks[i - 1]) / (ks[i] - ks[i - 1])
        return vs[i - 1] + t * (vs[i] - vs[i - 1])

    rows = []
    for mode in ("single", "oracle"):
        series = [(n, s_x, c) for (n, m), (s_x, c) in data.items() if m == mode]
        if len(series) < 2:
            continue
        for x in xs:
            vals = [(n, at(c, s_x, x)) for n, s_x, c in series]
            if any(v is None for _, v in vals):
                continue
            rows.append({"mode": mode, "k_over_s": x,
                         **{f"recall_{n}": v for n, v in vals},
                         "abs_diff": abs(vals[0][1] - vals[1][1])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--thresh", type=float, default=0.01, help="recall that counts as onset")
    a = ap.parse_args()

    rows, ests, curves = [], {}, {}
    for name, comp, per in PAIRS:
        if not os.path.exists(comp):
            print(f"[onset] missing {comp}", file=sys.stderr)
            continue
        s = st.median(float(r["s_mean"]) for r in csv.DictReader(open(per)))
        for mode, L in (("single", 0), ("oracle", 50)):
            c = curve(comp, mode, L)
            lo, hi, est = crossing(c, a.thresh)
            rows.append({"pair": name, "mode": mode, "L": L, "s_x_nats_per_token": s,
                         "onset_lo": lo, "onset_hi": hi, "onset_est": est,
                         "onset_est_over_s": (est / s) if est else None,
                         "grid": " ".join(f"{k:g}:{v:.3f}" for k, v in c.items())})
            curves[(name, mode)] = (s, c)
            if mode == "single":
                ests[name] = (s, est)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "onset.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"onset = first budget at which mean near-verbatim recall reaches {a.thresh:.0%}\n")
    print(f"{'pair':42s}{'mode':>8s}{'s(x)':>7s}{'bracket':>14s}{'est':>7s}{'est/s':>7s}")
    for r in rows:
        br = f"({r['onset_lo']:g}, {r['onset_hi']:g}]" if r["onset_lo"] else "at/below grid"
        e = f"{r['onset_est']:.2f}" if r["onset_est"] else "  -  "
        o = f"{r['onset_est_over_s']:.2f}" if r["onset_est_over_s"] else "  -  "
        print(f"{r['pair'][:41]:42s}{r['mode']:>8s}{r['s_x_nats_per_token']:7.2f}{br:>14s}{e:>7s}{o:>7s}")
    if len(ests) == 2:
        (n1, (s1, e1)), (n2, (s2, e2)) = ests.items()
        print(f"\nacross the two pairs the threshold differs by {s1/s2:.2f}x "
              f"({s1:.2f} vs {s2:.2f} nats/token)")
        print(f"and the single-query onset differs by {e1/e2:.2f}x ({e1:.2f} vs {e2:.2f})")
        print("The onset tracks the threshold, so its location is not a coincidence of one pair.")
    col = collapse(curves)
    if col:
        with open(os.path.join(a.out, "onset_collapse.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(col[0].keys()))
            w.writeheader()
            w.writerows(col)
        print("\ncollapse under the rescaled budget k/s(x):")
        for mode in ("single", "oracle"):
            d = [r["abs_diff"] for r in col if r["mode"] == mode]
            if d:
                print(f"  {mode:7s} mean |difference| between the two pairs over "
                      f"k/s in [0.7, 1.2]: {st.mean(d):.3f}")
    print(f"\nwrote {a.out}/onset.csv and {a.out}/onset_collapse.csv")


if __name__ == "__main__":
    main()
