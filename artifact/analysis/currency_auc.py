"""The within-prompt AUC behind the introduction's "AUC 0.526 ... the answer's length at 0.537".

feat-087 computed it once, inside a scoring log, and wrote no CSV; two referee reports then asked for
the estimator and for an interval, because both numbers sit near chance and the difference between
them is 0.011. This recomputes it from the committed per-candidate file and adds the intervals.

ESTIMATOR. results/selection_candidates.csv holds the n=8 anchor draws of the 500 judged prompts, each
judged once against the same opponent (judge A, Qwen2.5-7B-Instruct, one randomised order per
candidate) and scored win > tie > loss. Within each prompt, take every pair of candidates the judge
placed differently; a statistic is concordant on a pair when the better-judged candidate has the
larger value, half when the two values are equal. The AUC is the concordant fraction pooled over all
such pairs. Statistics: the risky model's per-token log-likelihood (what anchored decoding tilts
toward), its summed log-likelihood, and the completion's length in risky-model tokens.

INTERVALS. Prompts are resampled with replacement (the pair is not the independent unit: the eight
candidates of one prompt share a judge context), and the length-minus-likelihood difference is
computed on the same resample, so it is paired.

No GPU, no judge.  Usage:  .venv/bin/python analysis/currency_auc.py --out results
"""
import argparse
import csv
import os
import random
from itertools import combinations

RANK = {"loss": 0, "tie": 1, "win": 2}
STATS = (("per-token log-likelihood under p_r", "logp_per_token"),
         ("summed log-likelihood under p_r", "logp_total"),
         ("completion length, tokens", "n_tokens"))


def per_prompt(rows):
    """prompt -> (pairs, {stat: concordant sum}) over the pairs the judge placed differently."""
    by = {}
    for r in rows:
        by.setdefault(r["prompt_id"], []).append(r)
    out = {}
    for p, cands in by.items():
        pairs, conc = 0, {c: 0.0 for _, c in STATS}
        for a, b in combinations(cands, 2):
            ra, rb = RANK[a["outcome"]], RANK[b["outcome"]]
            if ra == rb:
                continue
            hi, lo = (a, b) if ra > rb else (b, a)
            pairs += 1
            for _, c in STATS:
                x, y = float(hi[c]), float(lo[c])
                conc[c] += 1.0 if x > y else 0.5 if x == y else 0.0
        out[p] = (pairs, conc)
    return out


def pooled(pp, pids, col):
    n = sum(pp[p][0] for p in pids)
    return sum(pp[p][1][col] for p in pids) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="results/selection_candidates.csv")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=8702)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    pp = per_prompt(list(csv.DictReader(open(a.candidates, encoding="utf-8"))))
    pids = sorted(pp)
    point = {c: pooled(pp, pids, c) for _, c in STATS}
    rng = random.Random(a.seed)
    boots = {c: [] for _, c in STATS}
    diff = []
    for _ in range(a.reps):
        s = [pids[rng.randrange(len(pids))] for _ in pids]
        v = {c: pooled(pp, s, c) for _, c in STATS}
        for c in v:
            boots[c].append(v[c])
        diff.append(v["n_tokens"] - v["logp_per_token"])

    def ci(xs):
        xs = sorted(xs)
        return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs))]

    n_pairs = sum(pp[p][0] for p in pids)
    rows = []
    for name, c in STATS:
        lo, hi = ci(boots[c])
        rows.append(dict(statistic=name, column=c, auc=round(point[c], 6), lo95=round(lo, 6),
                         hi95=round(hi, 6), n_pairs=n_pairs, n_prompts=len(pids),
                         excludes_half="yes" if lo > 0.5 or hi < 0.5 else "no"))
    lo, hi = ci(diff)
    rows.append(dict(statistic="length minus per-token log-likelihood", column="difference",
                     auc=round(point["n_tokens"] - point["logp_per_token"], 6), lo95=round(lo, 6),
                     hi95=round(hi, 6), n_pairs=n_pairs, n_prompts=len(pids),
                     excludes_half="excludes zero" if lo > 0 or hi < 0 else "includes zero"))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "currency_auc.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"[auc] {r['statistic']:42s} {r['auc']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}]"
              f"  {r['excludes_half']}  ({r['n_pairs']} pairs, {r['n_prompts']} prompts)")
    print(f"[auc] wrote {path}")


if __name__ == "__main__":
    main()
