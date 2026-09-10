"""Plan v5: does the temperature elasticity survive matching on memoriser strength?

The decoder warps both logit vectors before the KL solve, so tau = 0.4 sharpens the risky model as
well as the anchor and the warped arm's unconstrained memoriser is the stronger one. A stronger
memoriser reaches the threshold at a lower budget, which biases the measured onset down and the
elasticity toward 0 -- the direction that argues for the constant-nats null.

This restricts both arms to the passages the memoriser reproduces in BOTH (k = -1 near-verbatim
recall >= --match-thresh), so unconstrained strength is matched by construction, and recomputes
each arm's onset, s(x) and the elasticity on that subset alone. Pre-registered in
results/onset_prediction_matched_strength.md. No GPU.

  .venv/bin/python analysis/matched_strength.py --out results
"""
import argparse, csv, math, os, random, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.split_robustness import crossing  # noqa: E402

# (pair, control sweep, control budget path, warped sweep, warped budget path, unmatched elasticity)
ARMS = [
    ("KL3M-520M tau 0.4",
     "output/phase5/fine_kl3m520m/composition.csv",
     "results/budget_path_kl3m-520m__mem._kl3m-520m.csv",
     "output/phase5/warp_t0.4_kl3m520m/composition.csv",
     "results/budget_path_recon_t0.4.csv", 0.72),
    ("Pleias-1.2B tau 0.4",
     "output/phase5/fine_pleias12b/composition.csv",
     "results/budget_path_pleias-1.2b__mem._pleias-1.2b.csv",
     "output/phase5/warp_t0.4_pleias12b/composition.csv",
     "results/budget_path_recon_p12b_t0.4.csv", 0.61),
]


def sweep(path, column="lcs_word"):
    """{prompt_id: {k: value}} for k > 0, plus the k = -1 unconstrained baseline per passage."""
    curve, base = {}, {}
    for r in csv.DictReader(open(path)):
        if r["mode"] != "single":
            continue
        k = float(r["k"])
        if k > 0:
            curve.setdefault(r["prompt_id"], {})[k] = float(r[column])
        elif k == -1.0:
            base[r["prompt_id"]] = float(r["nv_recall"])
    return curve, base


def s_by_pid(path):
    return {r["prompt_id"]: float(r["s_mean"]) for r in csv.DictReader(open(path))}


def onset_on(curve, pids, thresh):
    ks = sorted({k for p in pids for k in curve.get(p, ())})
    mean = {}
    for k in ks:
        v = [curve[p][k] for p in pids if k in curve.get(p, ())]
        if v:
            mean[k] = st.mean(v)
    return crossing(mean, thresh)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--match-thresh", type=float, default=0.7,
                    help="k=-1 near-verbatim recall required in BOTH arms (pre-registered 0.7)")
    ap.add_argument("--lcs-thresh", type=float, default=4.0)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()

    rows = []
    for label, cc, cbp, wc, wbp, unmatched in ARMS:
        c_curve, c_base = sweep(cc)
        w_curve, w_base = sweep(wc)
        c_s, w_s = s_by_pid(cbp), s_by_pid(wbp)
        pids = sorted(p for p in set(c_curve) & set(w_curve) & set(c_s) & set(w_s)
                      if c_base.get(p, 0) >= a.match_thresh and w_base.get(p, 0) >= a.match_thresh)
        if len(pids) < 10:
            print(f"[matched] {label}: only {len(pids)} matched passages, skipping", file=sys.stderr)
            continue

        def elasticity(sample):
            oc, ow = onset_on(c_curve, sample, a.lcs_thresh), onset_on(w_curve, sample, a.lcs_thresh)
            if oc is None or ow is None:
                return None
            sc = st.median(c_s[p] for p in sample)
            sw = st.median(w_s[p] for p in sample)
            ds = math.log(sw / sc)
            return (math.log(ow / oc) / ds, oc, ow, sc, sw) if abs(ds) > 1e-9 else None

        point = elasticity(pids)
        rng = random.Random(1234)
        draws = []
        for _ in range(a.boot):
            sample = [pids[rng.randrange(len(pids))] for _ in pids]
            e = elasticity(sample)
            if e is not None:
                draws.append(e[0])
        draws.sort()
        lo = draws[int(0.025 * len(draws))] if draws else None
        hi = draws[int(0.975 * len(draws))] if draws else None
        e, oc, ow, sc, sw = point
        rows.append(dict(arm=label, n_matched=len(pids), n_total=len(set(c_curve) & set(w_curve)),
                         match_thresh=a.match_thresh,
                         base_recall_control=round(st.mean(c_base[p] for p in pids), 4),
                         base_recall_warped=round(st.mean(w_base[p] for p in pids), 4),
                         base_recall_control_all=round(st.mean(c_base.values()), 4),
                         base_recall_warped_all=round(st.mean(w_base.values()), 4),
                         s_x_control=round(sc, 4), s_x_warped=round(sw, 4),
                         onset_control=round(oc, 4), onset_warped=round(ow, 4),
                         ratio_control=round(oc / sc, 4), ratio_warped=round(ow / sw, 4),
                         elasticity=round(e, 3),
                         elasticity_lo=round(lo, 3) if lo is not None else "",
                         elasticity_hi=round(hi, 3) if hi is not None else "",
                         no_crossing_pct=round(100 * (a.boot - len(draws)) / a.boot, 1),
                         elasticity_unmatched=unmatched))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "matched_strength.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("matched on the passages the memoriser reproduces in BOTH arms "
          f"(k=-1 nv-recall >= {a.match_thresh}); elasticity 0 = a fixed number of nats, "
          "1 = proportional to s(x)\n")
    print(f"{'arm':22s}{'n':>5s}{'k=-1 ctl/warp':>16s}{'elasticity':>12s}{'95% CI':>18s}"
          f"{'unmatched':>11s}{'nocross':>9s}")
    for r in rows:
        ci = (f"[{r['elasticity_lo']:+.2f}, {r['elasticity_hi']:+.2f}]"
              if r["elasticity_lo"] != "" else "")
        print(f"{r['arm'][:21]:22s}{r['n_matched']:>5d}"
              f"{r['base_recall_control']:>8.3f}/{r['base_recall_warped']:<7.3f}"
              f"{r['elasticity']:>+12.2f}{ci:>18s}{r['elasticity_unmatched']:>+11.2f}"
              f"{r['no_crossing_pct']:>8.1f}%")
    print("\nwrote " + os.path.join(a.out, "matched_strength.csv"))


if __name__ == "__main__":
    main()
