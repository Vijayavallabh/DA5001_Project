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
            v = [y for _, y in vals]
            # plan v5: spread over EVERY pair. This was abs(vals[0]-vals[1]), which silently
            # compared only the first two series and would have hidden a third pair disagreeing.
            rows.append({"mode": mode, "k_over_s": x, "n_pairs": len(v),
                         **{f"recall_{n}": y for n, y in vals},
                         "spread": max(v) - min(v),
                         "sd": st.pstdev(v) if len(v) > 1 else 0.0})
    return rows


def load_pairs(path):
    """Pair set as data, not code: TSV of name<TAB>composition_summary.csv<TAB>budget_path.csv,
    with an optional fourth field naming the pair's tokenizer (used by analysis/onset_units.py)."""
    if not path or not os.path.exists(path):
        return PAIRS
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) not in (3, 4):
            raise SystemExit(f"[onset] {path}: expected 3 or 4 tab-separated fields, got {len(f)}: {line}")
        out.append(tuple(f[:3]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--thresh", type=float, default=0.01, help="recall that counts as onset")
    ap.add_argument("--pairs-file", default="results/onset_pairs.tsv",
                    help="TSV of name/composition_summary.csv/budget_path.csv; falls back to PAIRS")
    a = ap.parse_args()

    rows, ests, curves = [], {}, {}
    for name, comp, per in load_pairs(a.pairs_file):
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
    usable = {n: (s_x, e) for n, (s_x, e) in ests.items() if e}
    if len(usable) >= 2:
        ss = [s_x for s_x, _ in usable.values()]
        es = [e for _, e in usable.values()]
        print(f"\nacross {len(usable)} pairs s(x) spans {min(ss):.2f}-{max(ss):.2f} nats/token "
              f"({max(ss)/min(ss):.2f}x) and the single-query onset spans {min(es):.2f}-{max(es):.2f} "
              f"({max(es)/min(es):.2f}x)")
        ratios = [e / s_x for s_x, e in usable.values()]
        print(f"onset/s(x) per pair: " + ", ".join(f"{n}={e/s_x:.2f}" for n, (s_x, e) in usable.items()))
        print(f"  range {min(ratios):.2f}-{max(ratios):.2f}, sd {st.pstdev(ratios):.3f}"
              if len(ratios) > 1 else "")
    col = collapse(curves)
    if col:
        with open(os.path.join(a.out, "onset_collapse.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(col[0].keys()))
            w.writeheader()
            w.writerows(col)
        print("\ncollapse under the rescaled budget k/s(x):")
        for mode in ("single", "oracle"):
            d = [r["spread"] for r in col if r["mode"] == mode]
            if d:
                n = col[0]["n_pairs"]
                print(f"  {mode:7s} mean spread across {n} pairs over "
                      f"k/s in [0.7, 1.2]: {st.mean(d):.3f}")
    print(f"\nwrote {a.out}/onset.csv and {a.out}/onset_collapse.csv")


if __name__ == "__main__":
    main()
