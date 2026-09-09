"""Plan v5: a confidence interval on the extraction onset, by resampling passages.

The onset is the budget at which MEAN near-verbatim recall over a corpus first reaches a threshold.
Near that budget the mean is carried by one or two passages out of a hundred, so the crossing is
noisy: on one pair recall reads 0.003, 0.001, 0.012, 0.010 at k = 3.0, 3.1, 3.2, 3.3. A point
estimate from the first crossing is therefore not enough to separate competing predictions that sit
0.2 nats apart, and this script says how much it is not enough by.

Method: resample the passages with replacement, recompute the mean recall curve over the same k
grid, re-interpolate the first crossing, and report the percentile interval. Passages are the unit
of resampling because they are the independent replicates -- the same 100 works are decoded at
every budget, so the curve's points are correlated through them.

Reads a composition.csv (per-passage recall at each budget) and writes <out>/onset_ci.csv.
No GPU.

Usage:
  .venv/bin/python analysis/onset_ci.py --comp output/phase5/fine_pleias350m/composition.csv \
      --s-x 3.554 --label "Pleias-350M + mem. Pleias-350M" --out results
"""
import argparse
import collections
import csv
import os
import random
import statistics as st


def crossing(curve, thresh):
    """First budget at which the curve reaches `thresh`, linearly interpolated in the bracket."""
    prev = None
    for k in sorted(curve):
        if curve[k] >= thresh:
            if prev is None:
                return k
            lo, hi = curve[prev], curve[k]
            return prev + (k - prev) * (thresh - lo) / (hi - lo) if hi > lo else k
        prev = k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--s-x", type=float, required=True, help="vacuity threshold s(x), nats/token")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--mode", default="single")
    ap.add_argument("--thresh", type=float, default=0.01)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    by_pid = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.comp)):
        if r["mode"] != a.mode or float(r["k"]) <= 0:
            continue
        by_pid[r["prompt_id"]][float(r["k"])] = float(r["nv_recall"])
    pids = sorted(by_pid)
    ks = sorted({k for d in by_pid.values() for k in d})
    if not pids:
        raise SystemExit(f"[onset-ci] no rows in {a.comp} for mode={a.mode}")

    point = crossing({k: st.mean(by_pid[p][k] for p in pids if k in by_pid[p]) for k in ks}, a.thresh)

    boot = []
    for _ in range(a.boot):
        sample = [pids[rng.randrange(len(pids))] for _ in pids]
        curve = {}
        for k in ks:
            vals = [by_pid[p][k] for p in sample if k in by_pid[p]]
            if vals:
                curve[k] = st.mean(vals)
        c = crossing(curve, a.thresh)
        if c is not None:
            boot.append(c)
    boot.sort()
    n_none = a.boot - len(boot)
    lo = boot[int(0.025 * len(boot))] if boot else None
    hi = boot[int(0.975 * len(boot))] if boot else None

    row = {
        "pair": a.label, "mode": a.mode, "thresh": a.thresh, "n_passages": len(pids),
        "k_grid": " ".join(f"{k:g}" for k in ks),
        "onset_point": round(point, 4) if point else "",
        "onset_lo95": round(lo, 4) if lo else "", "onset_hi95": round(hi, 4) if hi else "",
        "boot_no_crossing_pct": round(100 * n_none / a.boot, 1),
        "s_x": a.s_x,
        "ratio_point": round(point / a.s_x, 4) if point else "",
        "ratio_lo95": round(lo / a.s_x, 4) if lo else "",
        "ratio_hi95": round(hi / a.s_x, 4) if hi else "",
    }
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "onset_ci.csv")
    exists = os.path.exists(path)
    rows = list(csv.DictReader(open(path))) if exists else []
    rows = [r for r in rows if r.get("pair") != a.label] + [row]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{a.label}: {len(pids)} passages, threshold {a.thresh}")
    print(f"  onset      {row['onset_point']}  95% CI [{row['onset_lo95']}, {row['onset_hi95']}]")
    print(f"  onset/s(x) {row['ratio_point']}  95% CI [{row['ratio_lo95']}, {row['ratio_hi95']}]")
    print(f"  {row['boot_no_crossing_pct']}% of bootstrap curves never reach the threshold")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
