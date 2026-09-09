"""Plan v5: does the derived requirement r(x) screen an INDIVIDUAL work, where s(x) does not?

The onset section reports a population boundary and says plainly that it is a poor per-work screen,
citing a weak correlation between s(x) and recall (-0.15, -0.26). But s(x) is not what the
derivation says governs a single work: Eq. (req) says the budget a decoder needs before it can pay
for x at all is

    r(x) = s_s(x) - s_r(x),

so at a budget k the works that can leak are exactly those with r(x) <= k. That is a per-work
CLASSIFICATION claim and it is testable on data we already have: 100 passages per pair with
per-passage recall at six budgets each.

For every (pair, budget) cell this computes the AUC of three scores as predictors of "this passage
leaked at this budget", higher score = more likely to leak:

    -r(x)      the derivation's own quantity
    -s_s(x)    the anchor's surprisal alone, the normaliser the earlier reading used
     s_r(x)    memorisation strength alone

If the derivation is right, -r(x) beats both. If -s_s(x) ties it, the extra term buys nothing and
the derivation is not earning its keep per work. No GPU: reads committed CSVs only.

Writes <out>/per_work_screen.csv (one row per pair, budget and score) and _summary.csv.
Usage:
  .venv/bin/python analysis/per_work_screen.py --out results
"""
import argparse
import csv
import os
import statistics as st

PAIRS = [
    ("TinyComma-1.8B + mem. Llama-3.1-8B", "output/phase4/fine_tc/composition.csv"),
    ("Comma-7B + mem. Comma-7B", "output/phase4/fine_comma/composition.csv"),
]


def auc(scores, labels):
    """Rank-based AUC with ties at 0.5. Returns None when one class is absent."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--per-work", default="results/onset_theory_per_work.csv")
    ap.add_argument("--thresh", type=float, default=0.01,
                    help="per-passage recall that counts as 'this work leaked'")
    ap.add_argument("--mode", default="single")
    a = ap.parse_args()

    theory = {}
    for r in csv.DictReader(open(a.per_work)):
        theory[(r["pair"], r["prompt_id"])] = (
            float(r["s_safe"]), float(r["s_risky"]), float(r["requirement"]))

    rows = []
    for pair, comp in PAIRS:
        if not os.path.exists(comp):
            print(f"[screen] missing {comp}, skipping {pair}")
            continue
        by_k = {}
        for r in csv.DictReader(open(comp)):
            if r["mode"] != a.mode:
                continue
            key = (pair, r["prompt_id"])
            if key not in theory:
                continue
            by_k.setdefault(float(r["k"]), []).append((key, float(r["nv_recall"])))
        for k in sorted(by_k):
            items = by_k[k]
            labels = [rec > a.thresh for _, rec in items]
            n_leak = sum(labels)
            scores = {
                "-r(x)": [-theory[key][2] for key, _ in items],
                "-s_safe(x)": [-theory[key][0] for key, _ in items],
                "s_risky(x)": [theory[key][1] for key, _ in items],
            }
            for name, sc in scores.items():
                rows.append({"pair": pair, "k": k, "mode": a.mode, "n": len(items),
                             "n_leaked": n_leak, "score": name,
                             "auc": auc(sc, labels),
                             # the derivation's own decision rule, not just a ranking
                             "frac_leaked_with_r_le_k": (
                                 sum(1 for (key, rec) in items if theory[key][2] <= k and rec > a.thresh)
                                 / max(1, sum(1 for (key, _) in items if theory[key][2] <= k))),
                             "frac_leaked_with_r_gt_k": (
                                 sum(1 for (key, rec) in items if theory[key][2] > k and rec > a.thresh)
                                 / max(1, sum(1 for (key, _) in items if theory[key][2] > k)))})

    if not rows:
        print("[screen] nothing to do")
        return 1
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per_work_screen.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for name in ("-r(x)", "-s_safe(x)", "s_risky(x)"):
        vals = [r["auc"] for r in rows if r["score"] == name and r["auc"] is not None]
        if vals:
            summary.append({"score": name, "n_cells": len(vals), "auc_mean": st.mean(vals),
                            "auc_min": min(vals), "auc_max": max(vals)})
    with open(os.path.join(a.out, "per_work_screen_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"per-work screen: does a score predict WHICH passage leaks at a given budget?")
    print(f"(AUC over {len({(r['pair'], r['k']) for r in rows})} pair-budget cells, "
          f"leak = single-query recall > {a.thresh})\n")
    print(f"{'score':14s}{'cells':>7s}{'mean AUC':>10s}{'min':>7s}{'max':>7s}")
    for s in summary:
        print(f"{s['score']:14s}{s['n_cells']:7d}{s['auc_mean']:10.3f}{s['auc_min']:7.3f}{s['auc_max']:7.3f}")
    print(f"\n{'pair':34s}{'k':>6s}{'leaked':>8s}{'AUC -r':>8s}{'AUC -s_s':>10s}")
    for pair, _ in PAIRS:
        for k in sorted({r["k"] for r in rows if r["pair"] == pair}):
            g = {r["score"]: r for r in rows if r["pair"] == pair and r["k"] == k}
            f = lambda n: f"{g[n]['auc']:.3f}" if g[n]["auc"] is not None else "  -  "
            print(f"{pair[:33]:34s}{k:6g}{g['-r(x)']['n_leaked']:8d}{f('-r(x)'):>8s}{f('-s_safe(x)'):>10s}")
    print(f"\nwrote {a.out}/per_work_screen.csv and _summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
