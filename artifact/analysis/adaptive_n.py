"""Adaptive (early-stopped) selection from the committed reward cache (results/adaptive_n_note.md).

Draw in rank order, stop at the first draw whose reward >= tau and serve it; if none of the first
n_max qualifies, serve the argmax of those n_max. Proposition 1 needs only "serve one of the draws",
so the rule is certified at log n_max however early it stops. tau is a global quantile q of the
rank-0 reward distribution. No GPU; --picks writes the served ranks of the two judged rules and their
compute-matched fixed-n arms for analysis/levels_pass.py.

Usage: .venv/bin/python analysis/adaptive_n.py --out results
"""
import argparse
import csv
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QS = (0.50, 0.75, 0.90, 0.95, 0.99)
JUDGED = ((64, 0.75), (64, 0.90))


def load(path):
    R, W = {}, {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        R.setdefault(r["prompt_id"], {})[int(r["rank"])] = float(r["reward"])
        W.setdefault(r["prompt_id"], {})[int(r["rank"])] = int(r["n_words"])
    return ({p: [v[i] for i in range(len(v))] for p, v in R.items()},
            {p: [v[i] for i in range(len(v))] for p, v in W.items()})


def rule(rs, tau, nmax):
    """(served rank, draws used)."""
    for i in range(nmax):
        if rs[i] >= tau:
            return i, i + 1
    return max(range(nmax), key=lambda i: rs[i]), nmax


def argmax_n(rs, n):
    return max(range(n), key=lambda i: rs[i])


def boot(v, rng, B=10000):
    n = len(v)
    ms = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards", default="results/selection_rewards64.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    R, W = load(a.rewards)
    pids = sorted(R)
    r0 = sorted(R[p][0] for p in pids)
    rng = random.Random(0)
    rows, picks = [], {}
    for nmax in (8, 64):
        for q in QS:
            tau = r0[min(len(r0) - 1, int(q * len(r0)))]
            served = {p: rule(R[p], tau, nmax) for p in pids}
            used = [served[p][1] for p in pids]
            nm = max(1, round(sum(used) / len(used)))
            d_full = [R[p][served[p][0]] - R[p][argmax_n(R[p], nmax)] for p in pids]
            d_cm = [R[p][served[p][0]] - R[p][argmax_n(R[p], nm)] for p in pids]
            lf, hf = boot(d_full, rng)
            lc, hc = boot(d_cm, rng)
            rows.append(dict(n_max=nmax, q=q, tau=round(tau, 4), mean_draws=round(sum(used) / len(used), 3),
                             median_draws=st.median(used), stop_first=round(sum(u == 1 for u in used) / len(used), 4),
                             served_reward=round(sum(R[p][served[p][0]] for p in pids) / len(pids), 4),
                             minus_fixed_nmax=round(sum(d_full) / len(d_full), 4), lo_nmax=round(lf, 4), hi_nmax=round(hf, 4),
                             matched_n=nm, minus_fixed_matched=round(sum(d_cm) / len(d_cm), 4),
                             lo_matched=round(lc, 4), hi_matched=round(hc, 4),
                             empty_served=round(sum(W[p][served[p][0]] == 0 for p in pids) / len(pids), 4),
                             certificate_nats=round(__import__("math").log(nmax), 4)))
            if (nmax, q) in JUDGED:
                picks[f"ada{nmax}_q{int(q * 100)}"] = {p: served[p][0] for p in pids}
                picks[f"fix{nm}_for_q{int(q * 100)}"] = {p: argmax_n(R[p], nm) for p in pids}
    path = os.path.join(a.out, "adaptive_n.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    for name, d in picks.items():
        with open(os.path.join(a.out, f"adaptive_picks_{name}.csv"), "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["prompt_id", "rank"])
            w.writerows(sorted(d.items()))
    print(f"wrote {path} and {len(picks)} pick files")


if __name__ == "__main__":
    main()
