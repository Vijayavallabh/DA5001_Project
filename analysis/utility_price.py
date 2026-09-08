"""Plan v5: how many nats does the decoder actually spend per nat of utility it buys?

Theorem 1 says a per-trajectory budget K lower-bounds the Cramer rate function of any bounded
utility under the safe model: K >= Lambda*_s(E_q[U]). That is a statement about the OPTIMAL policy.
This script measures the same two quantities for the decoder we actually have, and the gap between
them is the approximation error the theorem leaves open.

  Lambda*_s(u) = sup_lambda [ lambda*u - log E_{p_s}[e^{lambda U}] ]

U is the judge's verdict on one generation against the unconstrained model, scored win=1, tie=0.5,
loss=0, so U is bounded and its law under p_s is the three-point distribution measured on the
anchor-only arm. That makes the moment generating function exact rather than estimated by sampling,
and the supremum a one-dimensional concave maximisation.

The measured spend is the mean realised sequence divergence, sum_t c_t, read from the trajectory
logs -- not the budget cap K, which the decoder does not exhaust.

Both the utility gain and the rate function are bootstrapped over judged pairs, because a rate
function evaluated near the safe model's own mean is extremely sensitive to that mean.

Writes <out>/utility_price.csv. No GPU.

Usage:
  .venv/bin/python analysis/utility_price.py --out results
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import statistics as st

SCORE = {"win": 1.0, "tie": 0.5, "loss": 0.0}


def three_point(win_pct, loss_pct):
    w = win_pct / 100.0
    l = loss_pct / 100.0
    return w, max(0.0, 1.0 - w - l), l


def mean_u(d):
    w, t, l = d
    return w * SCORE["win"] + t * SCORE["tie"] + l * SCORE["loss"]


def rate(d_safe, u, lo=-200.0, hi=200.0, iters=200):
    """Lambda*_s(u) by ternary search on the concave objective. 0 when u is the safe mean."""
    w, t, l = d_safe

    def f(lam):
        mgf = w * math.exp(lam * 1.0) + t * math.exp(lam * 0.5) + l
        return lam * u - math.log(mgf)

    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if f(m1) < f(m2):
            lo = m1
        else:
            hi = m2
    return max(0.0, f((lo + hi) / 2.0))


def resample(d, n, rng):
    """One bootstrap draw of a three-point distribution from n judged pairs."""
    w, t, l = d
    cw = ct = 0
    for _ in range(n):
        x = rng.random()
        if x < w:
            cw += 1
        elif x < w + t:
            ct += 1
    return cw / n, ct / n, (n - cw - ct) / n


def spends(run_dir, k):
    out = []
    for c in ("neutral", "factual", "creative"):
        p = os.path.join(run_dir, f"trajectories_k{k}_{c}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            out.append(json.loads(line)["aggregate"]["total_spend"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--summary", default="results/utility_v4_summary.csv")
    ap.add_argument("--runs", default="output/phase2/conc_all")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = {(r["decoder"], r["k"]): r for r in csv.DictReader(open(a.summary))}
    anchor = rows[("anchor only", "0.0")]
    d_safe = three_point(float(anchor["win_pct"]), float(anchor["loss_pct"]))
    n_safe = int(anchor["n_judged"])
    u_safe = mean_u(d_safe)

    out = []
    for k in ("0.5", "1", "3", "5", "10", "20"):
        key = ("KL", k if "." in k else f"{k}.0")
        if key not in rows:
            continue
        r = rows[key]
        d_q = three_point(float(r["win_pct"]), float(r["loss_pct"]))
        n_q = int(r["n_judged"])
        u = mean_u(d_q)
        sp = spends(a.runs, k)
        if not sp:
            continue
        lam = rate(d_safe, u)
        boot = []
        for _ in range(a.boot):
            ds = resample(d_safe, n_safe, rng)
            dq = resample(d_q, n_q, rng)
            boot.append(rate(ds, mean_u(dq)))
        boot.sort()
        lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
        spend = st.mean(sp)
        out.append({
            "k": float(k), "budget_K": float(k) * 200, "mean_spend_nats": round(spend, 2),
            "n_trajectories": len(sp), "n_judged": n_q,
            "u_safe": round(u_safe, 4), "u_decoder": round(u, 4),
            "utility_gain": round(u - u_safe, 4),
            "lambda_star_nats": round(lam, 5),
            "lambda_star_lo": round(lo, 5), "lambda_star_hi": round(hi, 5),
            "spend_over_lambda_star": (round(spend / lam, 1) if lam > 1e-9 else ""),
        })

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "utility_price.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"safe model utility E_ps[U] = {u_safe:.4f} (win/tie/loss "
          f"{d_safe[0]:.3f}/{d_safe[1]:.3f}/{d_safe[2]:.3f}, n={n_safe})\n")
    print(f"{'k':>5s}{'spend':>9s}{'E_q[U]':>9s}{'gain':>8s}{'Lambda*':>10s}"
          f"{'95% CI':>18s}{'spend/Lambda*':>15s}")
    for r in out:
        ci = f"[{r['lambda_star_lo']:.4f},{r['lambda_star_hi']:.4f}]"
        print(f"{r['k']:>5g}{r['mean_spend_nats']:>9.1f}{r['u_decoder']:>9.4f}"
              f"{r['utility_gain']:>+8.4f}{r['lambda_star_nats']:>10.5f}{ci:>18s}"
              f"{str(r['spend_over_lambda_star']):>15s}")
    print(f"\nwrote {a.out}/utility_price.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
