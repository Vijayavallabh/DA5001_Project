"""Event-level audit (feat-007, C4): realised log-likelihood-ratio tails per k.

L(y) = sum_t log p_theta(y_t) / p_s(y_t) is the privacy-loss analogue of the KL certificate: its expectation is
E[Z] <= K, but single outputs can exceed K. Reads results/per_trajectory.csv (released logs, feat-002) and any
number of E1 output directories (--sweep name=dir) and writes results/llr_tails.csv with, per (source, k, split):
quantiles of L, the fraction with L > K together with a 95% anytime-valid confidence sequence (PrPl-EB, feat-007)
on that fraction, the largest excess L - K (empirical Delta_max lower bound), and single-step bursts; and
figures/llr_tails.{pdf,png}.

Usage: .venv/bin/python analysis/llr_tails.py --sweep plain=output/sweep_plain --out results
"""
import argparse, csv, glob, math, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.reanalyze_logs import FNAME, load  # noqa: E402
from dap.stats import anytime_valid_cs  # noqa: E402


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def rows_from(source, recs_by_k_split):
    rows = []
    for (k, split), recs in sorted(recs_by_k_split.items()):
        R = list(recs.values())
        if k <= 0:
            continue  # baselines have no budget (k=-1) or no served deviation (k=0)
        K = R[0]["K"]
        L = [r["L"] for r in R]
        ind = [1.0 if l > K else 0.0 for l in L]
        lo, hi = anytime_valid_cs(ind, alpha=0.05)
        bursts = [r["Lmax"] for r in R if r["Lmax"] > -math.inf]
        rows.append(dict(source=source, k=k, K=K, split=split, n_traj=len(R),
                         L_mean=round(st.mean(L), 2), L_p50=round(q(L, .5), 2), L_p90=round(q(L, .9), 2), L_p95=round(q(L, .95), 2),
                         L_p99=round(q(L, .99), 2), L_max=round(max(L), 2),
                         n_L_gt_K=int(sum(ind)), frac_L_gt_K=round(sum(ind) / len(ind), 4), frac_L_gt_K_cs95_lo=round(lo, 4), frac_L_gt_K_cs95_hi=round(hi, 4),
                         max_excess_L_minus_K=round(max(l - K for l in L), 2), frac_L_gt_half_K=round(sum(l > K / 2 for l in L) / len(L), 4),
                         step_boost_p95=round(q(bursts, .95), 3), step_boost_max=round(max(bursts), 3),
                         Z_mean=round(st.mean(r["Z"] for r in R), 2)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--released", default="results/per_trajectory.csv")
    ap.add_argument("--sweep", action="append", default=[], help="name=dir of an E1 output directory (repeatable)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()

    data = {}  # source -> {(k, split): {key: rec}}
    if os.path.exists(args.released):
        by = {}
        for r in csv.DictReader(open(args.released)):
            rec = dict(k=float(r["k"]), K=float(r["K"]), L=float(r["L"]), Lmax=float(r["Lmax"]), Z=float(r["Z"]))
            by.setdefault((rec["k"], r["split"]), {})[(r["prompt_id"], r["seed"])] = rec
        data["released"] = by
    for spec in args.sweep:
        name, d = spec.split("=", 1)
        by = {}
        for path in sorted(glob.glob(os.path.join(d, "trajectories_k*_*.jsonl"))):
            mm = FNAME.search(path)
            by[(float(mm["k"]), mm["split"])] = load(path)
        data[name] = by

    rows = []
    for source, by in data.items():
        rows += rows_from(source, by)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "llr_tails.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['source']} k={r['k']:g} {r['split']}: L>K in {r['n_L_gt_K']}/{r['n_traj']} = {r['frac_L_gt_K']} (CS95 [{r['frac_L_gt_K_cs95_lo']}, {r['frac_L_gt_K_cs95_hi']}]), max excess {r['max_excess_L_minus_K']}, step max {r['step_boost_max']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    ax = axes[0]
    for source, by in data.items():
        for (k, split), recs in sorted(by.items()):
            if k <= 0 or split not in ("attack_train",):
                continue
            K = next(iter(recs.values()))["K"]
            xs = sorted(r["L"] / K for r in recs.values())
            ys = [(i + 1) / len(xs) for i in range(len(xs))]
            ax.plot(xs, ys, lw=1.2, label=f"k={k:g} ({source}, n={len(xs)})")
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("realised log-likelihood ratio L(y) / K")
    ax.set_ylabel("empirical CDF (attack_train)")
    ax.set_xlim(0, max(1.3, ax.get_xlim()[1]))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax = axes[1]
    for source in data:
        A = [r for r in rows if r["source"] == source and r["split"] == "attack_train"]
        if not A:
            continue
        ks = [r["k"] for r in A]
        ax.errorbar(ks, [100 * r["frac_L_gt_K"] for r in A],
                    yerr=[[100 * (r["frac_L_gt_K"] - r["frac_L_gt_K_cs95_lo"]) for r in A], [100 * (r["frac_L_gt_K_cs95_hi"] - r["frac_L_gt_K"]) for r in A]],
                    fmt="o-", capsize=3, label=f"P[L(y) > K], 95% anytime-valid CS ({source})")
    from matplotlib.ticker import NullFormatter
    ax.set_xscale("log")
    ks_all = sorted({r["k"] for r in rows})
    ax.set_xticks(ks_all)
    ax.set_xticklabels([f"{k:g}" for k in ks_all])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("per-token budget k")
    ax.set_ylabel("% of trajectories with L(y) > K")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(args.figures, exist_ok=True)
    fig.savefig(os.path.join(args.figures, "llr_tails.pdf"))
    fig.savefig(os.path.join(args.figures, "llr_tails.png"), dpi=150)
    print("wrote", os.path.join(args.out, "llr_tails.csv"), os.path.join(args.figures, "llr_tails.pdf"))


if __name__ == "__main__":
    main()
