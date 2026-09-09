"""Plan v5 / feat-060: score the truncation run against its pre-registration.

For a fixed corpus, target length in tokens and characters-per-token are the same variable: the
same passage is 276 tokens under a four-character tokenizer and 580 under a two-character one. The
truncation run breaks that link by decoding only the first N target tokens with the tokenizer held
fixed, so its onset answers which of the two moved the KL3M pairs above the vacuity threshold.

Scored on `lcs_word`, an absolute word count. `nv_recall` divides by reference length, so halving a
target roughly doubles it; it is reported here for completeness and explicitly is not the criterion.

Writes <out>/truncation_score.csv. No GPU.

Usage:
  .venv/bin/python analysis/score_truncation.py --out results \
    --trunc output/phase5/trunc276_kl3m520m/composition.csv \
    --trunc-budget-path results/budget_path_kl3m520m_trunc276.csv
"""
import argparse, collections, csv, os, random, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.split_robustness import crossing  # noqa: E402  (returns None with no bracket)

# the two pre-registered readings, from results/onset_prediction_trunc276.md, committed at 449766b
READINGS = {"decode-step count": 0.87, "tokenizer itself": 1.03}
REFERENCE = {  # lcs_word >= 4 onset ratios of the untruncated pairs
    "four-character pairs": (0.866, 0.893),
    "KL3M pairs, untruncated": (1.032, 1.155),
}


def per_passage(path, column, mode="single"):
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["mode"] == mode and float(r["k"]) > 0:
            by[r["prompt_id"]][float(r["k"])] = float(r[column])
    return by


def point_and_ci(by_pid, thresh, s_x, boot=4000, seed=1234):
    pids = sorted(by_pid)
    ks = sorted({k for d in by_pid.values() for k in d})
    mean_curve = {k: st.mean(by_pid[p][k] for p in pids if k in by_pid[p]) for k in ks}
    point = crossing(mean_curve, thresh)
    rng = random.Random(seed)
    draws = []
    for _ in range(boot):
        sample = [pids[rng.randrange(len(pids))] for _ in pids]
        c = {}
        for k in ks:
            v = [by_pid[p][k] for p in sample if k in by_pid[p]]
            if v:
                c[k] = st.mean(v)
        o = crossing(c, thresh)
        if o is not None:
            draws.append(o)
    draws.sort()
    lo = draws[int(0.025 * len(draws))] if draws else None
    hi = draws[int(0.975 * len(draws))] if draws else None
    return point, lo, hi, 100 * (boot - len(draws)) / boot, mean_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunc", default="output/phase5/trunc276_kl3m520m/composition.csv")
    ap.add_argument("--trunc-budget-path", default="results/budget_path_kl3m520m_trunc276.csv")
    ap.add_argument("--full", default="output/phase5/fine_kl3m520m/composition.csv")
    ap.add_argument("--full-budget-path", default="results/budget_path_kl3m-520m__mem._kl3m-520m.csv")
    ap.add_argument("--lcs-thresh", type=float, default=4.0)
    ap.add_argument("--nv-thresh", type=float, default=0.01)
    ap.add_argument("--out", default="results")
    # the same scoring serves any two-arm comparison against these bands (feat-062 reuses it
    # for the Phi pair), so the row labels and the file name are arguments, not constants.
    ap.add_argument("--trunc-label", default="truncated to 276 tokens")
    ap.add_argument("--full-label", default="full 580 tokens")
    ap.add_argument("--out-name", default="truncation_score.csv")
    a = ap.parse_args()

    rows = []
    for label, comp, bp in ((a.trunc_label, a.trunc, a.trunc_budget_path),
                            (a.full_label, a.full, a.full_budget_path)):
        if not os.path.exists(comp):
            print(f"[trunc] missing {comp}", file=sys.stderr)
            continue
        s_x = st.median(float(r["s_mean"]) for r in csv.DictReader(open(bp)))
        for metric, col, thresh, primary in (("lcs_word", "lcs_word", a.lcs_thresh, True),
                                             ("nv_recall", "nv_recall", a.nv_thresh, False)):
            o, lo, hi, nocross, _ = point_and_ci(per_passage(comp, col), thresh, s_x)
            rows.append({"run": label, "metric": metric, "primary": primary, "threshold": thresh,
                         "s_x": round(s_x, 4),
                         "onset": round(o, 4) if o else "",
                         "ci_lo": round(lo, 4) if lo else "", "ci_hi": round(hi, 4) if hi else "",
                         "ratio": round(o / s_x, 4) if o else "",
                         "ratio_lo": round(lo / s_x, 4) if lo else "",
                         "ratio_hi": round(hi / s_x, 4) if hi else "",
                         "boot_no_crossing_pct": round(nocross, 1)})
    if not rows:
        raise SystemExit("[trunc] nothing to score")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, a.out_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'run':26s}{'metric':11s}{'thr':>6s}{'onset':>8s}{'95% CI':>16s}{'ratio':>8s}"
          f"{'ratio CI':>16s}{'nocross':>9s}")
    for r in rows:
        if r["onset"] == "":
            print(f"{r['run']:26s}{r['metric']:11s}{r['threshold']:6g}{'never crosses':>48s}")
            continue
        star = " *" if r["primary"] else "  "
        ci = f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]"
        rci = f"[{r['ratio_lo']:.2f},{r['ratio_hi']:.2f}]"
        print(f"{r['run']:26s}{r['metric']:11s}{r['threshold']:6g}{r['onset']:8.3f}"
              f"{ci:>16s}{r['ratio']:8.3f}{rci:>16s}{r['boot_no_crossing_pct']:8.1f}%{star}")

    prim = [r for r in rows if r["primary"] and r["run"].startswith("truncated") and r["onset"] != ""]
    if prim:
        got = prim[0]["ratio"]
        print(f"\nprimary result: truncated ratio {got:.3f} on lcs_word >= {a.lcs_thresh:g}")
        for name, (lo, hi) in REFERENCE.items():
            inside = lo <= got <= hi
            print(f"  {name:26s} {lo:.3f}-{hi:.3f}   {'INSIDE' if inside else 'outside'}")
        near = min(READINGS, key=lambda n: abs(READINGS[n] - got))
        print(f"\n  pre-registered readings: " +
              ", ".join(f"{n} ~{v:.2f}" for n, v in READINGS.items()))
        print(f"  closest: {near} (predicted ~{READINGS[near]:.2f}, measured {got:.3f}, "
              f"|error| {abs(READINGS[near] - got):.3f})")
    print(f"\nwrote {a.out}/{a.out_name}")


if __name__ == "__main__":
    main()
