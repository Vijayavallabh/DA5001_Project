"""feat-148: the compute-matched comparison with the judge removed.

`analysis/compute_matched.py` answers "at the meter's serving cost, which mechanism gives more
utility?" on the JUDGED axis, and the answer there is a loss for selection
(`-0.0395 [-0.0720, -0.0065]`, the paper's largest concession). After the judge panel split on the
headline difference (`results/onset_prediction_frontier_judge.md`), a comparison that rests on one
judge is worth less than it was, and the same question can be asked with no judge at all.

It can be asked on TriviaQA, the one task where both mechanisms have been run at the one anchor a
meter shares a vocabulary with (`results/verifiable_metered_tqa.csv`, `T_max = 24`).

THE COST MODEL is `analysis/serving_cost.py`'s, unchanged, with one simplification that favours the
METER: majority vote needs no scorer at all (the "score" is a mode over n strings), so

    selection, n      n * P_anchor * (L_p + T)
    metered, any k    (P_anchor + P_risky) * (L_p + T)

and the ratio is `n * P_anchor / (P_anchor + P_risky)`, independent of the token counts, which
cancel exactly because both arms answer the same 500 questions under the same `T_max`.

THE POINT OF THE ARM is not to win. The metered decoder's accuracy is constant in its serving cost
-- it runs both models at every step whatever `k` is -- so its whole accuracy range is available at
one cost, and it wins the compute-matched comparison outright. What the arm measures is the
CERTIFICATE it needs to do that, which is the paper's thesis with the judge removed.

Reads:  results/verifiable_metered_tqa.csv
Writes: <out>/compute_matched_judgefree.csv

No GPU, no judge, no generation.

Usage:
  .venv/bin/python analysis/compute_matched_judgefree.py --out results
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.serving_cost import P_ANCHOR, P_RISKY  # noqa: E402

SRC = "results/verifiable_metered_tqa.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.src, encoding="utf-8")))
    sel = [r for r in rows if r["mechanism"].startswith("selection")]
    # k=-1 is the UNCONSTRAINED risky model: the baseline the whole paper measures against,
    # not a metered arm. Its certificate field reads 0 because it has no budget, not
    # because it is cheap, and including it would credit the meter with a free 0.618.
    met = [r for r in rows if r["mechanism"] == "metered decoder" and r["arm"] != "k=-1"]
    assert sel and met, a.src

    met_cost = P_ANCHOR + P_RISKY          # per served token, every k
    parity_n = met_cost / P_ANCHOR         # the n at which selection costs the meter's compute

    # The meter's accuracy is constant in cost, so "the best metered arm at or below this cost" is
    # simply its best arm -- but we report the certificate that arm needs, which is not constant.
    best = max(met, key=lambda r: float(r["acc"]))

    out = []
    for r in sel:
        n = int(r["arm"].split("=")[1])
        cost = n * P_ANCHOR / met_cost
        out.append(dict(
            arm=r["arm"], n=n,
            cost_vs_metered=round(cost, 4),
            selection_acc=float(r["acc"]),
            selection_certificate_nats=float(r["certificate_nats"]),
            best_metered_acc=float(best["acc"]),
            best_metered_arm=best["arm"],
            best_metered_certificate_nats=float(best["certificate_nats"]),
            best_metered_realised_nats=float(best["realised_nats"]),
            accuracy_gap=round(float(best["acc"]) - float(r["acc"]), 4),
            # n=1 has selected nothing and carries a zero certificate; a ratio against it is
            # not large, it is undefined, and printing a number there would be a divide-by-epsilon
            # dressed as a measurement.
            certificate_ratio=("" if float(r["certificate_nats"]) == 0 else
                               round(float(best["certificate_nats"])
                                     / float(r["certificate_nats"]), 1)),
        ))

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "compute_matched_judgefree.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"[cmjf] metered serving cost is constant in k: both models every step, "
          f"{met_cost:.4f}B params per served token")
    print(f"[cmjf] selection reaches that cost at n = {parity_n:.2f}")
    for r in out:
        print(f"[cmjf] n={r['n']:3d}  cost {r['cost_vs_metered']:6.3f}x  "
              f"acc {r['selection_acc']:.3f} for {r['selection_certificate_nats']:.3f} nats  "
              f"vs best metered {r['best_metered_acc']:.3f} ({r['best_metered_arm']}) for "
              f"{r['best_metered_certificate_nats']:.0f} certified / "
              f"{r['best_metered_realised_nats']:.2f} realised")
    print(f"[cmjf] wrote {p}")
    print("[cmjf] READING: the metered decoder wins the compute-matched comparison on this axis "
          "outright, and it needs a certificate of "
          f"{best['certificate_nats']} nats for a 24-token answer to do it.")


if __name__ == "__main__":
    main()
