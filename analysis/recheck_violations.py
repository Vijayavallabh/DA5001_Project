"""Recompute per-query budget violations from a composition run's queries.jsonl under the right semantics
(KL decoder: Z <= max(0, B) + 1e-3; pathwise decoder: R <= max(0, B) + 1e-3), and report the maximum overshoot.
Usage: .venv/bin/python analysis/recheck_violations.py --queries output/phase2/comp8b_pathwise/queries.jsonl --constraint pathwise
"""
import argparse, json
from collections import defaultdict

EPS = 1e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--constraint", choices=["kl", "pathwise"], required=True)
    args = ap.parse_args()
    field = "R" if args.constraint == "pathwise" else "Z"
    n, viol, worst = 0, 0, float("-inf")
    per_k = defaultdict(lambda: [0, 0, float("-inf")])
    for line in open(args.queries):
        q = json.loads(line)
        if q["k"] <= 0:
            continue
        n += 1
        over = q[field] - max(0.0, q["B"])
        worst = max(worst, over)
        per_k[q["k"]][0] += 1
        per_k[q["k"]][2] = max(per_k[q["k"]][2], over)
        if over > EPS:
            viol += 1
            per_k[q["k"]][1] += 1
    print(f"{args.constraint}: {n} budgeted queries, {viol} violations of {field} <= max(0,B)+{EPS}, largest overshoot {worst:.6f}")
    for k in sorted(per_k):
        c, v, w = per_k[k]
        print(f"  k={k:g}: {c} queries, {v} violations, largest overshoot {w:.6f}")


if __name__ == "__main__":
    main()
