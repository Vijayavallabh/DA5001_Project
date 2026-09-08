"""Plan v5: join measured onsets to derived predictions and test the two competing hypotheses.

Two readings of where extraction begins:

  H_const   onset = c * s_s(x) for a universal constant c (the "0.89" pairs 1 and 2 share)
  H_deriv   onset = r(x) = s_s(x) - s_r(x)                (Eq. (req))

Across different anchors these are hard to separate, because s_s and s_r move together. The
graded-memoriser LADDER separates them: rungs share one anchor, so s_s is identical by construction
and only s_r varies. H_const then predicts every rung onsets at the same budget; H_deriv predicts
the onset falls as memorisation strengthens. They differ in the SIGN of the trend.

This script scores both hypotheses on whatever pairs are present, reporting per-pair absolute error
in nats and, for any set of rungs sharing an anchor, the sign of the measured trend against the
sign each hypothesis requires.

Reads results/onset.csv (measured, from analysis/onset.py) and results/onset_theory.csv (predicted,
from analysis/onset_theory.py). No GPU.

Usage:
  .venv/bin/python analysis/onset_ladder.py --out results
"""
import argparse
import csv
import os
import statistics as st


def load(path, key):
    if not os.path.exists(path):
        raise SystemExit(f"[ladder] missing {path}")
    return {r[key]: r for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--onset", default="results/onset.csv")
    ap.add_argument("--theory", default="results/onset_theory.csv")
    ap.add_argument("--const", type=float, default=None,
                    help="the constant coefficient to test; default = mean measured ratio")
    a = ap.parse_args()

    theory = load(a.theory, "pair")
    measured = {}
    for r in csv.DictReader(open(a.onset)):
        if r["mode"] == "single" and r.get("onset_est"):
            measured[r["pair"]] = (float(r["s_x_nats_per_token"]), float(r["onset_est"]))

    # match the two files' pair labels: onset.csv uses "memorised", onset_theory.csv uses "mem."
    def norm(s):
        return s.lower().replace("memorised", "mem").replace("mem.", "mem").replace(" ", "")
    tkey = {norm(k): k for k in theory}

    rows = []
    for pair, (s_x, onset) in sorted(measured.items()):
        t = theory.get(tkey.get(norm(pair), ""))
        if not t:
            print(f"[ladder] no prediction for {pair!r}, skipping")
            continue
        rows.append({"pair": pair, "s_safe": float(t["s_safe_median"]),
                     "s_risky": float(t["s_risky_median"]),
                     "temperature": t.get("temperature", "1.0"),
                     "measured_onset": onset, "measured_ratio": onset / s_x,
                     "pred_deriv": float(t["req_q25"])})
    if not rows:
        raise SystemExit("[ladder] no pairs with both a measurement and a prediction")

    const = a.const if a.const is not None else st.mean(r["measured_ratio"] for r in rows)
    for r in rows:
        r["pred_const"] = const * r["s_safe"]
        r["err_deriv"] = abs(r["pred_deriv"] - r["measured_onset"])
        r["err_const"] = abs(r["pred_const"] - r["measured_onset"])

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "onset_ladder.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"constant coefficient under test: c = {const:.3f}\n")
    print(f"{'pair':38s}{'s_s':>6s}{'s_r':>6s}{'meas':>7s}{'deriv':>7s}{'const':>7s}"
          f"{'|e|dv':>7s}{'|e|ct':>7s}")
    for r in rows:
        print(f"{r['pair'][:37]:38s}{r['s_safe']:6.2f}{r['s_risky']:6.2f}{r['measured_onset']:7.2f}"
              f"{r['pred_deriv']:7.2f}{r['pred_const']:7.2f}{r['err_deriv']:7.2f}{r['err_const']:7.2f}")
    md = st.mean(r["err_deriv"] for r in rows)
    mc = st.mean(r["err_const"] for r in rows)
    print(f"\nmean |error|: derivation {md:.3f} nats, constant {mc:.3f} nats -> "
          f"{'derivation' if md < mc else 'constant'} wins by {abs(md-mc):.3f}")

    # rungs sharing an anchor: s_s equal to 2 dp
    groups = {}
    for r in rows:
        groups.setdefault(round(r["s_safe"], 2), []).append(r)
    for s_s, g in sorted(groups.items()):
        if len(g) < 2:
            continue
        g = sorted(g, key=lambda r: r["s_risky"])
        print(f"\nladder at s_s = {s_s:.2f} ({len(g)} rungs, anchor held fixed):")
        print(f"  {'s_r':>6s}{'measured onset':>16s}{'derivation':>12s}")
        for r in g:
            print(f"  {r['s_risky']:6.3f}{r['measured_onset']:16.2f}{r['pred_deriv']:12.2f}")
        rise = g[-1]["measured_onset"] - g[0]["measured_onset"]
        print(f"  onset changes by {rise:+.2f} nats as s_r rises from {g[0]['s_risky']:.3f} to "
              f"{g[-1]['s_risky']:.3f}")
        print(f"  H_const requires  0.00 (same anchor, same onset)")
        print(f"  H_deriv requires {g[0]['s_risky'] - g[-1]['s_risky']:+.2f} (onset = s_s - s_r)")
    print(f"\nwrote {a.out}/onset_ladder.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
