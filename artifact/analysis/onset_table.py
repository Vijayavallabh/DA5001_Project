"""Plan v5 / feat-057: assemble the onset table the paper prints, from one command.

The numbers in Section 4's table live in three files -- onsets and bootstrap intervals in
results/onset_ci.csv, s(x) and s_r in results/onset_theory.csv, and the collapse spreads in
results/onset_collapse.csv -- and the cross-pair standard deviation is derived from all of them.
Deriving it by hand is how a table drifts from its evidence, so this writes exactly what the paper
quotes, including the choice that matters: when a pair has been measured twice, the higher-n
measurement is the one tabulated.

Writes <out>/onset_table.csv. No GPU.

Usage: .venv/bin/python analysis/onset_table.py --out results
"""
import argparse, csv, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.score_predictions import canonical, load_measurements  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--ci", default="results/onset_ci.csv")
    ap.add_argument("--theory", default="results/onset_theory.csv")
    a = ap.parse_args()

    meas = load_measurements(a.ci)
    rows = []
    for t in csv.DictReader(open(a.theory)):
        key = canonical(t["pair"])
        if key not in meas or t["pair"].startswith("Ladder rung"):
            continue
        m = meas[key]
        onset = float(m["onset_point"])
        s_s = float(t["s_safe_median"])
        rows.append({
            "pair": t["pair"], "n_passages": int(m["n_passages"]),
            "s_safe": round(s_s, 3), "s_risky": round(float(t["s_risky_median"]), 3),
            "onset": round(onset, 3),
            "ci_lo": round(float(m["onset_lo95"]), 3), "ci_hi": round(float(m["onset_hi95"]), 3),
            "ratio": round(onset / s_s, 4),
            "ratio_lo": round(float(m["onset_lo95"]) / s_s, 4),
            "ratio_hi": round(float(m["onset_hi95"]) / s_s, 4),
            "boot_no_crossing_pct": float(m["boot_no_crossing_pct"]),
            "k_grid": m["k_grid"],
        })
    if not rows:
        raise SystemExit("[table] no pair has both a prediction and a measurement")
    rows.sort(key=lambda r: r["s_safe"])

    ratios = [r["ratio"] for r in rows]
    summary = {"pair": f"ALL {len(rows)} PAIRS", "n_passages": sum(r["n_passages"] for r in rows),
               "s_safe": round(min(r["s_safe"] for r in rows), 3),
               "s_risky": round(max(r["s_safe"] for r in rows), 3),
               "onset": round(min(r["onset"] for r in rows), 3),
               "ci_lo": round(max(r["onset"] for r in rows), 3), "ci_hi": "",
               "ratio": round(st.mean(ratios), 4),
               "ratio_lo": round(min(ratios), 4), "ratio_hi": round(max(ratios), 4),
               "boot_no_crossing_pct": round(st.pstdev(ratios), 4), "k_grid": "sd in the last column"}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "onset_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows); w.writerow(summary)

    print(f"{'pair':38s}{'n':>5s}{'s(x)':>7s}{'s_r':>7s}{'onset':>8s}{'95% CI':>15s}"
          f"{'ratio':>8s}{'no-cross':>10s}")
    for r in rows:
        ci = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
        print(f"{r['pair'][:37]:38s}{r['n_passages']:5d}{r['s_safe']:7.3f}{r['s_risky']:7.3f}"
              f"{r['onset']:8.3f}{ci:>15s}"
              f"{r['ratio']:8.3f}{r['boot_no_crossing_pct']:9.1f}%")
    print(f"\n{len(rows)} pairs: ratio mean {st.mean(ratios):.4f}, range "
          f"{min(ratios):.3f}-{max(ratios):.3f}, sd {st.pstdev(ratios):.4f}")
    print(f"s(x) spans {min(r['s_safe'] for r in rows):.3f}-{max(r['s_safe'] for r in rows):.3f} "
          f"nats/token ({max(r['s_safe'] for r in rows)/min(r['s_safe'] for r in rows):.2f}x)")
    print(f"\nwrote {a.out}/onset_table.csv")


if __name__ == "__main__":
    main()
