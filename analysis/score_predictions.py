"""Plan v5 / feat-056: score the three onset rules the way a pre-registration demands.

Three rules have been on the table for where near-verbatim extraction begins:

  q25       the population onset is the 25th percentile of the per-work requirement r(x),
            calibrated on the first two pairs;
  P1        the onset is the median of r(x) = s_s(x) - s_r(x) itself, which follows from
            psi_t(1) = 0 on the geodesic and has nothing fitted in it;
  constant  the onset is c * s(x) for a single c, also calibrated on the first two pairs.

Two of the three were calibrated on pairs 1 and 2, so scoring them there measures the fit and not
the rule. Only pairs measured AFTER their prediction was committed are a test, and this script
reports those separately and leads with them. Predictions come from results/onset_theory.csv
(written before each sweep ran); measurements and their bootstrap intervals come from
results/onset_ci.csv, which also carries the passage count, so when a pair has been measured twice
the higher-n measurement is the one scored.

Writes <out>/prediction_scores.csv. No GPU.

Usage:
  .venv/bin/python analysis/score_predictions.py --out results \
    --calibrated-on "TinyComma-1.8B + mem. Llama-3.1-8B" "Comma-7B + mem. Comma-7B"
"""
import argparse, csv, os, re, statistics as st, sys


def canonical(name):
    """The two manifests spell a pair differently and a re-measurement adds an (n=...) tag."""
    s = name.lower().replace("memorised", "mem.")
    s = re.sub(r"\s*\(n=\d+\)\s*$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_measurements(path):
    """canonical pair -> row, keeping the measurement with the most passages."""
    best = {}
    for r in csv.DictReader(open(path)):
        if r["mode"] != "single" or not r["onset_point"]:
            continue
        k = canonical(r["pair"])
        n = int(r["n_passages"])
        if k not in best or n > int(best[k]["n_passages"]):
            best[k] = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--theory", default="results/onset_theory.csv")
    ap.add_argument("--ci", default="results/onset_ci.csv")
    ap.add_argument("--calibrated-on", nargs="+", required=True,
                    help="pairs the q25 and constant rules were fitted on; they are reported but "
                         "excluded from the held-out score")
    a = ap.parse_args()

    meas = load_measurements(a.ci)
    cal = {canonical(x) for x in a.calibrated_on}
    preds = [r for r in csv.DictReader(open(a.theory))
             if canonical(r["pair"]) in meas and not r["pair"].startswith("Ladder rung")]
    if not preds:
        raise SystemExit(f"[score] no pair in {a.theory} has a measurement in {a.ci}")

    # the constant was fitted as the mean measured ratio on the calibration pairs; refit it here
    # from those same pairs so the number in the paper and the number scored cannot drift apart.
    cal_ratios = [float(meas[canonical(r["pair"])]["onset_point"]) / float(r["s_safe_median"])
                  for r in preds if canonical(r["pair"]) in cal]
    c = st.mean(cal_ratios) if cal_ratios else float("nan")

    rows = []
    for r in preds:
        m = meas[canonical(r["pair"])]
        y = float(m["onset_point"])
        lo, hi = float(m["onset_lo95"]), float(m["onset_hi95"])
        rules = {"q25 of r(x)": float(r["req_q25"]),
                 "P1: median s_s - s_r": float(r["pred_onset_median"]),
                 f"constant {c:.3f}*s(x)": c * float(r["s_safe_median"])}
        for rule, p in rules.items():
            rows.append({"pair": r["pair"], "rule": rule,
                         "held_out": canonical(r["pair"]) not in cal,
                         "n_passages": int(m["n_passages"]),
                         "s_safe": round(float(r["s_safe_median"]), 4),
                         "s_risky": round(float(r["s_risky_median"]), 4),
                         "measured": round(y, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "predicted": round(p, 4), "error": round(p - y, 4),
                         "abs_error": round(abs(p - y), 4),
                         "rel_error_pct": round(100 * (p - y) / y, 2),
                         "inside_ci": int(lo <= p <= hi),
                         "measured_ratio": round(y / float(r["s_safe_median"]), 4)})

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "prediction_scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rules = sorted({r["rule"] for r in rows})
    print(f"constant refitted on {len(cal_ratios)} calibration pair(s): c = {c:.4f}\n")
    for held in (True, False):
        sub = [r for r in rows if r["held_out"] is held]
        if not sub:
            continue
        print(f"--- {'HELD OUT (a test)' if held else 'CALIBRATION PAIRS (a fit, not a test)'} ---")
        print(f"{'pair':36s}{'n':>5s}{'measured':>10s}{'95% CI':>16s}"
              + "".join(f"{r.split(':')[0][:16]:>17s}" for r in rules))
        for pair in dict.fromkeys(r["pair"] for r in sub):
            pr = {r["rule"]: r for r in sub if r["pair"] == pair}
            any_ = next(iter(pr.values()))
            ci = f"[{any_['ci_lo']:.2f},{any_['ci_hi']:.2f}]"
            cells = "".join(
                f"{pr[r]['predicted']:11.2f}{'  in' if pr[r]['inside_ci'] else ' OUT':>6s}"
                for r in rules)
            print(f"{pair[:35]:36s}{any_['n_passages']:5d}{any_['measured']:10.3f}{ci:>16s}{cells}")
        print()
    print(f"{'rule':28s}{'held-out mean |err|':>21s}{'in CI':>8s}{'all mean |err|':>16s}{'in CI':>8s}")
    for rule in rules:
        h = [r for r in rows if r["rule"] == rule and r["held_out"]]
        al = [r for r in rows if r["rule"] == rule]
        hv = f"{st.mean(r['abs_error'] for r in h):.3f}" if h else "  -  "
        hc = f"{sum(r['inside_ci'] for r in h)}/{len(h)}" if h else " - "
        print(f"{rule:28s}{hv:>21s}{hc:>8s}"
              f"{st.mean(r['abs_error'] for r in al):16.3f}"
              f"{sum(r['inside_ci'] for r in al)}/{len(al):>7}")
    print(f"\nwrote {a.out}/prediction_scores.csv")


if __name__ == "__main__":
    sys.exit(main())
