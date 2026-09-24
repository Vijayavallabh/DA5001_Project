"""The realised divergence of majority vote, against the log n it is certified at (post hoc).

Proposition 1's log n is a bound. For majority vote the served law can be computed rather than
bounded, because the rule only looks at the extracted ANSWERS: given the answer sequence of the n
draws the served index is fixed (the earliest draw whose answer is modal), and the draws are i.i.d.,
so given its answer class a the served string is distributed as p_s(. | a). Hence

    q(y) / p_s(y) = q_ans(a(y)) / pi(a(y)),

where pi is the anchor's law over answer classes and q_ans the law of the served class. D_inf(q||p_s)
is the largest of those ratios and D_KL(q||p_s) = KL(q_ans || pi), both exactly. pi is estimated by the
64 cached draws per question (a plug-in, so this is descriptive); q_ans by simulating n draws from
it under the committed rule (analysis/selection_verifiable.py: majority): answers that parse vote,
ties go to the class that appears first -- uniform over the tied classes by exchangeability --
and a question whose n draws all fail to parse serves draw 0.

Usage: .venv/bin/python analysis/majority_realised_divergence.py --out results
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_verifiable import extract, extract_tqa  # noqa: E402

TASKS = {"gsm8k": ("output/phase5/verifiable/anchor_comma7b_n64.jsonl", extract),
         "triviaqa": ("output/phase5/verifiable/anchor_tqa_comma7b_n64.jsonl", extract_tqa)}
N_GRID = (2, 4, 8, 16, 32, 64)
SIMS = 20000


def served_class_law(pi, none_idx, n, rng):
    """Monte Carlo law of the served answer class under the committed majority rule."""
    counts = rng.multinomial(n, pi, size=SIMS).astype(float)
    voting = counts.copy()
    if none_idx is not None:
        voting[:, none_idx] = 0.0
    top = voting.max(axis=1, keepdims=True)
    tied = (voting == top) & (top > 0)
    w = tied / np.maximum(tied.sum(axis=1, keepdims=True), 1)
    if none_idx is not None:          # every draw failed to parse: draw 0 is served, a None string
        w[(top[:, 0] == 0), none_idx] = 1.0
    return w.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=64)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for task, (path, pick) in TASKS.items():
        by = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            by.setdefault(r["qid"], []).append(r["text"])
        qids = sorted(by)[: a.limit]
        per_n = {n: ([], []) for n in N_GRID}
        for q in qids:
            ans = [pick(t) for t in by[q]]
            c = Counter("\0NONE" if x is None else x for x in ans)
            keys = sorted(c)
            pi = np.array([c[k] for k in keys], float) / len(ans)
            none_idx = keys.index("\0NONE") if "\0NONE" in c else None
            for n in N_GRID:
                qa = served_class_law(pi, none_idx, n, rng)
                ratio = np.where(qa > 0, qa / pi, 0.0)
                d_inf = float(np.log(ratio.max()))
                kl = float(np.sum(np.where(qa > 0, qa * np.log(np.where(qa > 0, qa, 1) / pi), 0.0)))
                per_n[n][0].append(d_inf)
                per_n[n][1].append(kl)
        for n in N_GRID:
            dinf, kl = np.array(per_n[n][0]), np.array(per_n[n][1])
            rows.append(dict(task=task, n=n, log_n=round(math.log(n), 4),
                             kl_bound=round(math.log(n) - (n - 1) / n, 4),
                             dinf_median=round(float(np.median(dinf)), 4),
                             dinf_p90=round(float(np.percentile(dinf, 90)), 4),
                             dinf_max=round(float(dinf.max()), 4),
                             kl_median=round(float(np.median(kl)), 4),
                             kl_p90=round(float(np.percentile(kl, 90)), 4),
                             n_questions=len(qids)))
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "majority_realised_divergence.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", out)


if __name__ == "__main__":
    main()
