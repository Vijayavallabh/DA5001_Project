"""How much of the seed-word gradient survives the noise in the points it is fitted to?

Appendix~\\ref{app:onset} reports that the onset ratio falls with the words a fixed 20-token seed
buys the adversary, at Spearman -0.958, exact p = 0.0002 over nine pairs. That statistic treats each
pair's ratio as a point. It is not one: every pair is a single LoRA fine-tune at --seed 0, and the
seed ladders measure what re-training one of them does -- a standard deviation of 0.0450 in the
ratio for Pleias-1.2B on CopyBench and 0.0180 for KL3M-520M, both pairs OF this table and on this
table's own corpus (analysis/strength_ladder.py, results/onset_group_dispersion.csv).

So this re-draws the nine ratios under that measured noise and asks what the gradient does. The
answer is the honest qualification: the direction is not at risk and the magnitude is. This is a
sensitivity analysis and is POST HOC -- no band was committed on it -- which is why it is reported
as a range rather than as a verdict.

The null is exact. At n = 9 there are 362,880 permutations, so the permutation distribution of
Spearman's rho against the (tied) seed-word vector is enumerated once and reused for every draw;
an asymptotic p at this size would be the wrong tool and the paper does not use one elsewhere.

    .venv/bin/python analysis/seedword_gradient.py   ->  results/seedword_gradient.csv
"""
import argparse
import bisect
import csv
import itertools
import os
import random
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The two re-seed standard deviations measured on pairs of this table, on this table's corpus.
NOISE = {"pleias12b_copybench": 0.0450, "kl3m520m_copybench": 0.0180}


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    for pos, i in enumerate(order):
        rk[i] = pos + 1.0
    for val in set(v):                                   # average ties: seed words has two pairs
        idx = [i for i, u in enumerate(v) if u == val]
        if len(idx) > 1:
            m = sum(rk[i] for i in idx) / len(idx)
            for i in idx:
                rk[i] = m
    return rk


def rho(rw, rr):
    mw, mr = st.mean(rw), st.mean(rr)
    num = sum((a - mw) * (b - mr) for a, b in zip(rw, rr))
    den = (sum((a - mw) ** 2 for a in rw) * sum((b - mr) ** 2 for b in rr)) ** 0.5
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/seedword_gradient.csv")
    a = ap.parse_args()

    with open(os.path.join(ROOT, "results/onset_seed_words.csv")) as fh:
        rows = list(csv.DictReader(fh))
    w = [float(r["seed_words"]) for r in rows]
    r0 = [float(r["ratio"]) for r in rows]
    assert len(rows) == 9, f"expected the nine pairs, got {len(rows)}"

    rw = ranks(w)
    null = sorted(abs(rho(rw, list(p))) for p in itertools.permutations(range(1, len(w) + 1)))
    def pval(x):
        return (len(null) - bisect.bisect_left(null, abs(x) - 1e-12)) / len(null)

    obs = rho(rw, ranks(r0))
    out = [{"arm": "observed", "reseed_sd": "", "n": len(rows), "trials": "",
            "rho_median": round(obs, 4), "rho_lo5": "", "rho_hi95": "",
            "pct_p_under_05": "", "pct_p_under_01": "", "pct_sign_kept": "",
            "exact_p": round(pval(obs), 4)}]

    for label, sd in NOISE.items():
        rng = random.Random(a.seed)
        rr, ps = [], []
        for _ in range(a.trials):
            rr.append(rho(rw, ranks([x + rng.gauss(0, sd) for x in r0])))
            ps.append(pval(rr[-1]))
        rr.sort()
        q = lambda f: rr[int(f * len(rr))]
        out.append({
            "arm": f"reseed_noise_{label}", "reseed_sd": sd, "n": len(rows), "trials": a.trials,
            "rho_median": round(st.median(rr), 4), "rho_lo5": round(q(0.05), 4),
            "rho_hi95": round(q(0.95), 4),
            "pct_p_under_05": round(100 * sum(1 for p in ps if p < 0.05) / a.trials, 1),
            "pct_p_under_01": round(100 * sum(1 for p in ps if p < 0.01) / a.trials, 1),
            "pct_sign_kept": round(100 * sum(1 for x in rr if x < 0) / a.trials, 1),
            "exact_p": ""})

    # Is the gradient confounded with which fine-tunes converged? Report the spans, not a verdict.
    with open(os.path.join(ROOT, "results/onset_table.csv")) as fh:
        conv = {r["pair"]: r["converged"] for r in csv.DictReader(fh) if " + " in r["pair"]}
    sw = {r["pair"]: float(r["seed_words"]) for r in rows}
    for flag in ("yes", "no"):
        v = sorted(sw[p] for p, c in conv.items() if c == flag)
        out.append({"arm": f"seed_words_where_converged_{flag}", "reseed_sd": "", "n": len(v),
                    "trials": "", "rho_median": "", "rho_lo5": round(v[0], 2),
                    "rho_hi95": round(v[-1], 2), "pct_p_under_05": "", "pct_p_under_01": "",
                    "pct_sign_kept": "", "exact_p": ""})

    with open(os.path.join(ROOT, a.out), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wtr.writeheader()
        wtr.writerows(out)
    for r in out:
        print("  ".join(f"{k}={v}" for k, v in r.items() if v != ""))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
