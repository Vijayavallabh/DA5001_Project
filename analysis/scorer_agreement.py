"""feat-116, post hoc: how much of the 7B scorer's ranking does a 0.5B scorer reproduce?

A diagnostic, not a band. No band was committed on it and it must not be reported as one. It exists
because whatever the judged frontier says, the reader will ask *why* -- and the two reward caches
answer it without a judge: if the small scorer picks the same draw as the big one, any judged
difference is noise; if it picks a different one, the judged difference is the scorer's capability.

Writes <out>/compute_matched_scorer_agreement.csv. No GPU: two cached CSVs.

Usage:
  .venv/bin/python analysis/scorer_agreement.py --out results
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_scaling import load_rewards  # noqa: E402

GRID = (2, 4, 8, 16, 32, 64)


def spearman(a, b):
    """Rank correlation with average ranks for ties; exact enough for a diagnostic."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rewards-7b", default="results/selection_rewards64.csv")
    ap.add_argument("--rewards-small", default="results/selection_rewards64_qwen05b.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    r7, rs = load_rewards(a.rewards_7b), load_rewards(a.rewards_small)
    pids = sorted(set(r7) & set(rs))
    assert pids, "the two reward caches share no prompt"

    rows = []
    rho = [spearman(r7[p][:64], rs[p][:64]) for p in pids]
    rows.append(dict(quantity="Spearman(7B, 0.5B) over 64 candidates, mean within prompt", n="",
                     value=round(sum(rho) / len(rho), 4), n_prompts=len(pids),
                     note="how much of the big scorer's ordering the small one reproduces"))
    for n in GRID:
        same = sum(max(range(n), key=lambda j: r7[p][j]) == max(range(n), key=lambda j: rs[p][j])
                   for p in pids)
        rows.append(dict(quantity="the two scorers serve the same draw", n=n,
                         value=round(same / len(pids), 4), n_prompts=len(pids),
                         note=f"chance is {1 / n:.4f} if the two rankings were independent"))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "compute_matched_scorer_agreement.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"  {r['quantity']:56s} n={str(r['n']):<3} {r['value']:+.4f}   {r['note']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
