"""Proposition 3, measured step by step on logs already written.

The proposition says a bucket-metered decoder's realised spend is the cost of IMITATING the risky
model on the steps where the bucket is slack, not the cost of the utility it achieves. Two things
follow that the per-step logs can check exactly:

  (i) On a slack step the solver returns p* = p_r, so the charge a_t IS D_KL(p_r,t || p_s,t). The
      proposition's lower bound (1-beta) sum_t D_KL(p_r,t || p_s,t) is therefore MEASURED, not
      estimated: it is the sum of a_t over the steps the log marks risky-unchanged.
 (ii) That sum grows at a fixed rate per token, so the running spend is linear in the step index and
      its slope does not move with the budget. results/utility_price.csv shows the arm means
      saturating; this shows the mechanism, inside each trajectory, and separates the two regimes:
      below the imitation rate the meter binds and the decoder spends its whole allowance, above it
      the decoder spends the imitation cost and the rest of the certificate is unreachable.

A cross-trajectory regression of spend on generation LENGTH cannot test this: every trajectory in
these sweeps runs to T_max, so the length has no spread (p5 200, p95 211 tokens) and the fit is
uninformative by construction. The variation that does exist is across steps within a trajectory,
which is what this script uses.

Zero GPU: it reads `per_step_log` from the sweeps on disk.

Reads:  output/{sweep_plain,phase2/conc_all}/trajectories_k*_{neutral,creative,factual,attack_train}.jsonl
Writes: <out>/imitation_cost.csv, one row per (budget, prompt class)

Usage:  .venv/bin/python analysis/imitation_cost.py --out results
"""
import argparse
import csv
import glob
import json
import math
import os

CLASSES = {"ordinary": ("neutral", "creative", "factual"), "protected": ("attack_train",)}
DIRS = ("output/sweep_plain", "output/phase2/conc_all")


def ols(xs, ys):
    """slope, intercept, R^2 of y on x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return b, a, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def mean(v):
    v = list(v)
    return sum(v) / len(v)


def median(v):
    """Median over the finite values; a trajectory whose spend never moves has no R^2 to take."""
    v = sorted(x for x in v if isinstance(x, (int, float)) and math.isfinite(x))
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def scan(k, splits):
    """One row of evidence per trajectory: slack fraction, slack charge, realised spend, linearity."""
    rows = []
    for d in DIRS:
        for sp in splits:
            for path in glob.glob(os.path.join(d, f"trajectories_k{k}_{sp}.jsonl")):
                for line in open(path):
                    r = json.loads(line)
                    a, m, log = r["aggregate"], r["metadata"], r["per_step_log"]
                    if float(m.get("k", -1)) <= 0 or not log:
                        continue
                    # A step is slack iff the solver returned p_r unchanged. `bd` is the weight
                    # the solve puts on the risky model; dap/e1.py:205 counts a step free at
                    # bd >= 1 - 1e-6, and the assertion below holds this scan to that definition.
                    slack = [e for e in log if e.get("bd", 0.0) >= 1 - 1e-6]
                    forced = [e for e in log if e.get("bd", 0.0) <= 1e-6]
                    # The aggregate carries the same count, but only where its three counters add
                    # up to the trajectory: the older sweep_plain runs wrote them from a truncated
                    # buffer (a k=0.1 record reports 1 step in total for a 200-step trajectory), so
                    # the per-step log is the source of truth and the aggregate is a check when it
                    # is self-consistent.
                    tally = sum(a.get(c, 0) for c in
                                ("steps_forced_safe", "steps_active", "steps_risky_unchanged"))
                    if tally == len(log):
                        assert len(slack) == a["steps_risky_unchanged"], (path, len(slack))
                    b, _, r2 = ols([e["t"] for e in log], [e["cum_kl_spent"] for e in log])
                    # How concentrated is the spend? Proposition 5 says a policy on a budget that
                    # does not grow with the work must put essentially all of it on O(1) steps.
                    # This measures the opposite end: how many steps the deployed rule needs to
                    # cover 90% of what it spent, and what its top 1% of steps carry.
                    charges = sorted((e["a_t"] for e in log), reverse=True)
                    tot = sum(charges) or 1.0
                    top1 = sum(charges[:max(1, len(charges) // 100)]) / tot
                    run, n90 = 0.0, len(charges)
                    for i, c in enumerate(charges, 1):
                        run += c
                        if run >= 0.9 * tot:
                            n90 = i
                            break
                    rows.append(dict(
                        T=len(log), K=float(m["K"]),
                        beta=1 - len(slack) / len(log),
                        forced=len(forced) / len(log),
                        active=1 - (len(slack) + len(forced)) / len(log),
                        slack_charge=sum(e["a_t"] for e in slack),
                        imitation_rate=(sum(e["a_t"] for e in slack) / len(slack)) if slack else 0.0,
                        spend=a["total_spend"], slope=b, r2=r2,
                        top1pct_share=top1, steps_for_90pct=n90 / len(log)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    ks = sorted({os.path.basename(p).split("_")[1][1:]
                 for d in DIRS for p in glob.glob(os.path.join(d, "trajectories_k*_neutral.jsonl"))},
                key=float)
    out = []
    for cls, splits in CLASSES.items():
        for k in [x for x in ks if float(x) > 0]:
            rows = scan(k, splits)
            if len(rows) < 50:
                continue
            K = rows[0]["K"]
            rec = dict(
                prompt_class=cls, k=k, budget_K=round(K, 1), n_trajectories=len(rows),
                beta_binding_frac=round(sum(r["beta"] for r in rows) / len(rows), 5),
                # beta splits into the prefix-debt opening, where the anchor writes the token
                # outright, and the steps where the solve returns a strict interior blend -- the
                # latter is the "constraint active" fraction the released logs report.
                forced_safe_frac=round(sum(r["forced"] for r in rows) / len(rows), 5),
                active_frac=round(sum(r["active"] for r in rows) / len(rows), 5),
                # Arm means, so these reconcile exactly with results/utility_price.csv, which
                # summarises the same trajectories; only the per-trajectory fits are medians.
                imitation_rate_nats_per_token=round(mean(r["imitation_rate"] for r in rows), 4),
                realised_rate_nats_per_token=round(mean(r["spend"] / r["T"] for r in rows), 4),
                # Proposition 3's right-hand side, summed over the steps that certify it
                prop3_lower_bound_nats=round(mean(r["slack_charge"] for r in rows), 2),
                spend_nats=round(mean(r["spend"] for r in rows), 2),
                spend_over_cap=round(mean(r["spend"] for r in rows) / K, 4),
                top1pct_of_steps_share_of_spend=round(mean(r["top1pct_share"] for r in rows), 4),
                frac_of_steps_for_90pct_of_spend=round(mean(r["steps_for_90pct"] for r in rows), 4),
                median_cum_spend_vs_step_slope=round(median(r["slope"] for r in rows), 4),
                median_cum_spend_vs_step_r2=round(median(r["r2"] for r in rows), 4))
            out.append(rec)
            print(f"  {cls:9s} k={float(k):5g}  n={rec['n_trajectories']:5d}  "
                  f"beta={rec['beta_binding_frac']:.4f}  imit={rec['imitation_rate_nats_per_token']:.4f}"
                  f"  realised={rec['realised_rate_nats_per_token']:.4f} nats/tok  "
                  f"spend/cap={rec['spend_over_cap']:.3f}  R2={rec['median_cum_spend_vs_step_r2']:.4f}")

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "imitation_cost.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print(f"wrote {p} ({len(out)} rows)")


if __name__ == "__main__":
    main()
