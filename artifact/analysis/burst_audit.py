"""Burst audit (feat-021b, plan v2 C10): the banking rule is a token bucket with unbounded depth, so a trajectory that
spends little early can release the accumulated allowance in a burst. For every trajectory in the given log directories
this reports the largest spend inside any window of W consecutive decode steps, compared with the nominal W*k, and the
largest bank (budget_remaining) reached. Reads dap/e1.py trajectory JSONL (per_step_log a_t, k_t, budget_remaining).
Writes <out>/<prefix>.csv (per k and class: quantiles of the largest W-window spend / (W k) and of the bank).
Usage: .venv/bin/python analysis/burst_audit.py --logs output/h1_outputs output/sweep_plain --out results
"""
import argparse, csv, glob, json, os, statistics as st
from collections import defaultdict


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="burst_audit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W = args.window
    groups = defaultdict(list)
    for d in args.logs:
        for f in sorted(glob.glob(os.path.join(d, "trajectories_k*_*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                md, steps = r["metadata"], r["per_step_log"]
                k = md["k"]
                if k <= 0 or len(steps) < 2:
                    continue
                a = [s["a_t"] for s in steps]
                bank = [s["budget_remaining"] for s in steps]
                cs = [0.0]
                for v in a:
                    cs.append(cs[-1] + v)
                win = max(cs[t + W] - cs[t] for t in range(0, max(1, len(a) - W + 1))) if len(a) >= W else cs[-1]
                groups[(k, md["split"], os.path.basename(d))].append((win / (W * k), max(bank), sum(a) / max(1, len(a)) / k))
                groups[(k, "all", os.path.basename(d))].append((win / (W * k), max(bank), sum(a) / max(1, len(a)) / k))
    rows = []
    for (k, split, src), xs in sorted(groups.items()):
        bursts = [x[0] for x in xs]
        banks = [x[1] for x in xs]
        rows.append(dict(source=src, k=k, split=split, n=len(xs), window=W,
                         burst_ratio_median=round(st.median(bursts), 3), burst_ratio_p90=round(quantile(bursts, 0.9), 3), burst_ratio_p99=round(quantile(bursts, 0.99), 3), burst_ratio_max=round(max(bursts), 3),
                         pct_burst_gt_2x=round(100 * sum(b > 2 for b in bursts) / len(bursts), 2), pct_burst_gt_5x=round(100 * sum(b > 5 for b in bursts) / len(bursts), 2),
                         bank_max_median=round(st.median(banks), 2), bank_max_p90=round(quantile(banks, 0.9), 2), bank_max_max=round(max(banks), 2),
                         mean_spend_over_k=round(st.median([x[2] for x in xs]), 3)))
    with open(os.path.join(args.out, f"{args.prefix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["split"] == "all":
            print(f"[burst] {r['source']} k={r['k']:g} n={r['n']}: largest {W}-step spend / ({W}k) median {r['burst_ratio_median']} p99 {r['burst_ratio_p99']} max {r['burst_ratio_max']}; >2x in {r['pct_burst_gt_2x']}%; bank max median {r['bank_max_median']} max {r['bank_max_max']}")
    print("wrote", os.path.join(args.out, f"{args.prefix}.csv"))


if __name__ == "__main__":
    main()
