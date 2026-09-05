"""Concentrated certificate for the KL decoder (feat-020, plan v2 C9 / Proposition 3).

Under Anchored Decoding, M_t = sum_{i<=t} (r_i - a_i) is a martingale, where r_i = log p_theta(y_i)/p_s(y_i) is the
realised log-ratio of the sampled token and a_i = KL(p_theta,i || p_s,i) its conditional mean (the spend). Its
increments are bounded above by b_i = m_i - a_i (m_i = max_v log p_theta,i(v)/p_s,i(v)) and have conditional variance
var_i = Var_{p_theta,i}[log p_theta,i/p_s,i]. Freedman's inequality (one-sided form, Tropp 2011 Thm 1.1; time-uniform
Bernstein bounds of Howard et al. 2021) gives, for caps b >= max_i b_i and v >= sum_i var_i,
    P(exists t: M_t >= lambda and V_t <= v) <= exp(-lambda^2 / (2 (v + b lambda / 3))),
so P(L(y) >= K + lambda) <= delta(lambda) whenever Z <= K, which the decoder enforces. The decoder logs b_i and var_i
(a_patch/factory.py, fields m_t / var_ratio), so a deployer can certify delta(lambda) by capping V_t and b_i, and an
auditor can compare the certified tail with the empirical tail of M_T = L(y) - Z.

Reads trajectory JSONL files (dap/e1.py output) and writes <out>/concentration.csv (per trajectory: Z, L, M_T, V_T, b_T)
and <out>/concentration_summary.csv (per k and class: quantiles of V_T and b_T, certified delta(lambda) at caps set to
the 90th and 99th percentiles, and the empirical P(M_T >= lambda)).
Usage: .venv/bin/python analysis/concentration.py --logs output/phase2/kl_smoke --out results
"""
import argparse, csv, glob, json, math, os, statistics as st
from collections import defaultdict

LAMBDAS = (10.0, 25.0, 50.0, 100.0)


def delta_freedman(lam, v, b):
    return math.exp(-lam * lam / (2.0 * (v + b * lam / 3.0)))


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True, help="directories with trajectories_k*_*.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="concentration")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for d in args.logs:
        for f in sorted(glob.glob(os.path.join(d, "trajectories_k*_*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                md, ag, steps = r["metadata"], r["aggregate"], r["per_step_log"]
                if md["k"] <= 0 or not steps or steps[0].get("r_t") is None:
                    continue
                own = [s for s in steps if s.get("r_t") is not None]
                Z = sum(s["a_t"] for s in own)
                L = sum(s["r_t"] for s in own)
                V = sum(s["var_ratio"] for s in own)
                b = max(s["m_t"] - s["a_t"] for s in own)
                rows.append(dict(k=md["k"], K=md["K"], split=md["split"], prompt_id=md["prompt_id"], seed=md["seed"], constraint=md.get("constraint", "kl"),
                                 T=len(own), Z=round(Z, 3), L=round(L, 3), M=round(L - Z, 3), V=round(V, 3), b=round(b, 3),
                                 B=round(ag["final_budget"], 3), L_over_K=round(L / md["K"], 4)))
    if not rows:
        raise SystemExit("no trajectories with r_t/var_ratio fields found")
    with open(os.path.join(args.out, f"{args.prefix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["k"], r["constraint"], r["split"])].append(r)
        groups[(r["k"], r["constraint"], "all")].append(r)
    for (k, cons, split), R in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        Vs, bs, Ms = [r["V"] for r in R], [r["b"] for r in R], [r["M"] for r in R]
        srow = dict(k=k, constraint=cons, split=split, n=len(R), T_mean=round(st.mean(r["T"] for r in R), 1),
                    V_median=round(st.median(Vs), 2), V_p90=round(quantile(Vs, 0.9), 2), V_p99=round(quantile(Vs, 0.99), 2),
                    b_median=round(st.median(bs), 2), b_p90=round(quantile(bs, 0.9), 2), b_p99=round(quantile(bs, 0.99), 2), b_max=round(max(bs), 2),
                    M_mean=round(st.mean(Ms), 2), M_p95=round(quantile(Ms, 0.95), 2), M_max=round(max(Ms), 2),
                    L_gt_K_pct=round(100 * sum(r["L"] > r["K"] for r in R) / len(R), 2))
        for lam in LAMBDAS:
            srow[f"emp_P_M_ge_{lam:g}"] = round(sum(m >= lam for m in Ms) / len(R), 4)
            for q in (0.9, 0.99):
                v, b = quantile(Vs, q), quantile(bs, q)
                srow[f"delta_{lam:g}_cap_p{int(q*100)}"] = round(delta_freedman(lam, v, b), 4)
            srow[f"delta_{lam:g}_cap_max"] = round(delta_freedman(lam, max(Vs), max(bs)), 4)
        summary.append(srow)
    with open(os.path.join(args.out, f"{args.prefix}_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        if s["split"] == "all":
            print(f"[conc] k={s['k']:g} {s['constraint']} n={s['n']}: V median {s['V_median']} p90 {s['V_p90']}; b median {s['b_median']} p90 {s['b_p90']} max {s['b_max']}; "
                  f"M p95 {s['M_p95']} max {s['M_max']}; L>K {s['L_gt_K_pct']}%; "
                  f"delta(50) at p90 caps {s['delta_50_cap_p90']} (empirical P(M>=50) {s['emp_P_M_ge_50']}); delta(100) {s['delta_100_cap_p90']}")
    print("wrote", os.path.join(args.out, f"{args.prefix}.csv"))


if __name__ == "__main__":
    main()
