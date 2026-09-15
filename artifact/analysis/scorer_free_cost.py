"""feat-119: what the cheapest instance of the mechanism costs, and what it buys.

POST HOC, and labelled so throughout: no band was committed, because there is no new run. This is
arithmetic over quantities already measured -- the serving-cost model of feat-114/116/117 and the
judge-free accuracies of feat-101/118 -- joined into the one table neither of them could produce
alone. It is deliberately NOT named onset_prediction_*, for the same reason feat-115 was not: a
post-hoc re-analysis must not be able to inflate the pre-registration count.

The observation it exists to make: selection's serving cost is dominated by the reward model, and
majority vote does not have one. Self-consistency picks the modal answer over the same n anchor
draws, so the "scorer" is a regex and costs no forward pass. It carries the identical log n
certificate -- Proposition 1 assumes nothing about the score, and a mode is a score -- while costing
n P_anchor (L_p + T) against selection's n (P_anchor + P_scorer)(L_p + T).

Three limits are written into the output and belong wherever the numbers are quoted:

  1. POST HOC. No band; the cost column is arithmetic and the accuracies are feat-118's.
  2. It needs a CANONICAL ANSWER. There is no majority over free-form text, so this does not
     transfer to the judged workload, where every other comparison in the paper lives.
  3. No metered decoder was run on GSM8K. The `x metered` column is this paper's standard cost
     denominator (the 200-token workload of serving_cost.py), NOT a measured head-to-head on this
     task. What the judge-free column compares is each rule against the SAME anchor at n=1.

And the thing it must not be read as saying: majority vote does not reach the unconstrained risky
model, which scores 0.786 greedy on these problems against its 0.546. The claim is about the
mechanisms that carry a certificate, not about beating the model the certificate exists to bound.

Writes <out>/scorer_free_cost.csv. No GPU.

Usage:
  .venv/bin/python analysis/scorer_free_cost.py --out results
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.serving_cost import P_ANCHOR, P_RISKY, P_SCORER  # noqa: E402

GRID = (2, 4, 8, 16, 32, 64)
LIMITS = ("post hoc; needs a canonical answer; the ratio is the cost model's denominator, "
          "not a measured head-to-head on this task")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgefree", default="results/verifiable_scorer_scale.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    acc = {(r["task"], r["scorer"], int(r["n"])): float(r["acc"])
           for r in csv.DictReader(open(a.judgefree))}
    base = {t: acc[(t, "majority vote", 1)] for t in ("gsm8k", "triviaqa")}
    metered = P_ANCHOR + P_RISKY

    rows = []
    for rule, pf, label in (("majority vote", 0.0, "none"),
                            ("7.6B", P_SCORER, "Qwen2.5-7B")):
        for n in GRID:
            c = n * (P_ANCHOR + pf) / metered
            rows.append(dict(rule=rule, scorer=label, n=n,
                             nats_certified=round(math.log(n), 4),
                             cost_vs_metered=round(c, 3),
                             gsm8k_acc=acc[("gsm8k", rule, n)],
                             gsm8k_gain=round(acc[("gsm8k", rule, n)] - base["gsm8k"], 4),
                             triviaqa_acc=acc[("triviaqa", rule, n)],
                             triviaqa_gain=round(acc[("triviaqa", rule, n)] - base["triviaqa"], 4),
                             limits=LIMITS))
    rows.sort(key=lambda r: r["cost_vs_metered"])

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "scorer_free_cost.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    for r in rows:
        print(f"  {r['rule']:14s} n={r['n']:<3} {r['cost_vs_metered']:>7.2f}x   "
              f"gsm8k {r['gsm8k_acc']:.3f} ({r['gsm8k_gain']:+.4f})   "
              f"tqa {r['triviaqa_acc']:.3f} ({r['triviaqa_gain']:+.4f})")

    mv = {r["n"]: r for r in rows if r["rule"] == "majority vote"}
    rw = {r["n"]: r for r in rows if r["rule"] == "7.6B"}
    best_mv = max(mv.values(), key=lambda r: r["gsm8k_acc"])
    best_rw = max(rw.values(), key=lambda r: r["gsm8k_acc"])
    print(f"\n  best scorer-free cell: n={best_mv['n']} at {best_mv['cost_vs_metered']:.2f}x "
          f"for {best_mv['gsm8k_gain']:+.4f}")
    print(f"  best reward cell:      n={best_rw['n']} at {best_rw['cost_vs_metered']:.2f}x "
          f"for {best_rw['gsm8k_gain']:+.4f}")
    print(f"  ratio: {best_mv['gsm8k_gain'] / best_rw['gsm8k_gain']:.2f}x the gain for "
          f"{100 * best_mv['cost_vs_metered'] / best_rw['cost_vs_metered']:.1f}% of the cost")
    print(f"  LIMITS: {LIMITS}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
