"""Plan v5 / feat-045: is the collapse an artifact? Three named critiques, answered. Zero GPU.

A data collapse is only evidence if it survives the standard objections, so each block here
answers a specific one rather than reporting a fit.

  metric      Schaeffer et al. (NeurIPS 2023) showed apparent sharp behaviour is often manufactured
              by a thresholded metric. nv_recall counts spans of >= 20 words on a median reference
              of 49, so it is quantised. We repeat the collapse in lcs_word, which is continuous
              and already logged, and compare the disagreement as a fraction of each metric's range.

  threshold   the onset is defined as the budget where mean recall first reaches a threshold. If
              the result moves with that arbitrary choice, it is a definition and not a finding.

  normaliser  the derivation in analysis/onset_theory.py says the operative per-work quantity is
              the REQUIREMENT r(x) = s_safe(x) - s_risky(x), not the anchor surprisal s(x) alone.
              That is a falsifiable prediction about which rescaling collapses the curves best, so
              we ablate r against s_safe, s_risky and no rescaling at all.

Writes <out>/collapse_robustness.csv.
Usage: .venv/bin/python analysis/collapse_robustness.py --out results
"""
import argparse, bisect, csv, os, statistics as st, sys
from collections import defaultdict

PAIRS = [("TinyComma", "output/phase4/fine_tc/composition.csv", "results/budget_path.csv",
          "TinyComma-1.8B + mem. Llama-3.1-8B"),
         ("Comma7B", "output/phase4/fine_comma/composition.csv", "results/budget_path_comma7b.csv",
          "Comma-7B + mem. Comma-7B")]
GRID = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2)


def curve(path, col, mode="single"):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["mode"] == mode:
            by[float(r["k"])].append(float(r[col]))
    return {k: st.mean(v) for k, v in sorted(by.items())}


def interp(c, scale, x):
    ks = [k / scale for k in sorted(c)]
    vs = [c[k] for k in sorted(c)]
    if x <= ks[0] or x >= ks[-1]:
        return None
    i = bisect.bisect_left(ks, x)
    t = (x - ks[i - 1]) / (ks[i] - ks[i - 1])
    return vs[i - 1] + t * (vs[i] - vs[i - 1])


def disagreement(curves, scales, grid=GRID):
    """Mean |difference| between the two pairs on a common rescaled grid."""
    d = []
    for x in grid:
        vals = [interp(curves[n], scales[n], x) for n in curves]
        if all(v is not None for v in vals):
            d.append(abs(vals[0] - vals[1]))
    return (st.mean(d), len(d)) if d else (float("nan"), 0)


def onset(c, thresh):
    prev = None
    for k in sorted(c):
        if c[k] >= thresh:
            if prev is None:
                return k
            lo, hi = c[prev], c[k]
            return prev + (k - prev) * (thresh - lo) / (hi - lo) if hi > lo else k
        prev = k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    theory = defaultdict(list)
    tp = "results/onset_theory_per_work.csv"
    if os.path.exists(tp):
        for r in csv.DictReader(open(tp)):
            theory[r["pair"]].append((float(r["s_safe"]), float(r["s_risky"]),
                                      float(r["requirement"])))
    s_safe = {n: st.median(float(r["s_mean"]) for r in csv.DictReader(open(bp)))
              for n, _, bp, _ in PAIRS}
    rows = []

    # 1. metric artifact
    for col in ("nv_recall", "lcs_word"):
        curves = {n: curve(f, col) for n, f, _, _ in PAIRS}
        scale = max(max(c.values()) for c in curves.values())
        m, n_pts = disagreement(curves, s_safe)
        rows.append({"block": "metric", "setting": col, "value": m,
                     "value_relative_to_metric_range": m / scale if scale else float("nan"),
                     "n_grid": n_pts})

    # 2. threshold artifact
    curves = {n: curve(f, "nv_recall") for n, f, _, _ in PAIRS}
    for th in (0.002, 0.005, 0.01, 0.02, 0.05):
        os_ = {n: onset(curves[n], th) for n in curves}
        if all(v for v in os_.values()):
            ratios = [os_[n] / s_safe[n] for n in curves]
            rows.append({"block": "threshold", "setting": f"onset at {th:g}",
                         "value": max(ratios) - min(ratios),
                         "value_relative_to_metric_range": st.mean(ratios), "n_grid": len(ratios)})

    # 3. normaliser ablation -- the derivation's own prediction
    if theory:
        key = {"TinyComma": "TinyComma-1.8B + mem. Llama-3.1-8B",
               "Comma7B": "Comma-7B + mem. Comma-7B"}
        opts = {
            "raw (no rescaling)": {n: 1.0 for n in curves},
            "s_safe": s_safe,
            "requirement r = s_safe - s_risky":
                {n: st.median(c for _, _, c in theory[key[n]]) for n in curves},
        }
        for name, sc in opts.items():
            grid = (2.0, 2.4, 2.8, 3.2) if name.startswith("raw") else GRID
            m, n_pts = disagreement(curves, sc, grid)
            rows.append({"block": "normaliser", "setting": name, "value": m,
                         "value_relative_to_metric_range": float("nan"), "n_grid": n_pts})

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "collapse_robustness.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for block in ("metric", "threshold", "normaliser"):
        sub = [r for r in rows if r["block"] == block]
        if not sub:
            continue
        print(f"\n[{block}]")
        for r in sub:
            extra = (f"  ({r['value_relative_to_metric_range']:.3f} of the metric's range)"
                     if block == "metric" else
                     f"  (mean ratio {r['value_relative_to_metric_range']:.3f})"
                     if block == "threshold" else "")
            print(f"  {r['setting']:36s} {r['value']:.4f}{extra}")
    print(f"\nwrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
