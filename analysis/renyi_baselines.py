"""Do the four Renyi orders share one baseline, as the decoder's structure says they must?

output/phase4/renyi_renyi_{1_0,2,4,8} swept k in {1,3,5} with no k=-1 and no k=0 arm, against
Working Rules' mandatory-baselines requirement. This scores the arms that filled the gap against the
prediction committed in results/onset_prediction_renyi_baselines.md before they ran:

    at k=-1 the decoder serves the risky model alone and at k=0 the safe model alone, and
    a_patch/factory.py reads self.constraint ONLY in the branch that solves under a budget,
    so all four orders must reproduce output/phase4/fine_tc_base exactly.

Exactly, not approximately. These are the same models on the same passages with the same seed, so
any difference at all is a difference in the code path, not noise -- which is why this compares the
stored decimals rather than applying a tolerance. The arms are SAMPLED, so a constraint object that
consumed the RNG differently would move the draw without touching the mixing weights; that is the
failure this is built to catch and it would show up here as a difference confined to oracle, which
re-samples per window.

    .venv/bin/python analysis/renyi_baselines.py   ->  results/renyi_baselines.csv
"""
import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = "output/phase4/fine_tc_base"
ORDERS = {"renyi:1.0": "output/phase4/renyi_renyi_1_0_base",
          "renyi:2": "output/phase4/renyi_renyi_2_base",
          "renyi:4": "output/phase4/renyi_renyi_4_base",
          "renyi:8": "output/phase4/renyi_renyi_8_base"}
KEYS = ("nv_recall_mean", "nv_recall_median", "lcs_word_mean", "lcs_word_max")


def arms(d):
    """(mode, L, k) -> the stored strings, so a comparison is of decimals and not of floats."""
    path = os.path.join(ROOT, d, "composition_summary.csv")
    if not os.path.exists(path):
        raise SystemExit(f"[renyi] {path} missing; run scripts/run_renyi_baselines.sh")
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if float(r["k"]) <= 0:
                out[(r["mode"], r["L"], float(r["k"]))] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/renyi_baselines.csv")
    a = ap.parse_args()

    ref = arms(REFERENCE)
    rows, disagreements = [], 0
    for order, d in ORDERS.items():
        got = arms(d)
        shared = sorted(set(ref) & set(got), key=lambda t: (t[2], t[0], t[1]))
        if not shared:
            raise SystemExit(f"[renyi] {d} shares no baseline arm with {REFERENCE}")
        for key in shared:
            mode, L, k = key
            same = all(ref[key][c] == got[key][c] for c in KEYS)
            disagreements += 0 if same else 1
            rows.append({"order": order, "mode": mode, "L": L, "k": k,
                         "nv_recall": got[key]["nv_recall_mean"],
                         "lcs_word": got[key]["lcs_word_mean"],
                         "reference_nv_recall": ref[key]["nv_recall_mean"],
                         "reference_lcs_word": ref[key]["lcs_word_mean"],
                         "identical_to_reference": "yes" if same else "NO",
                         "violations": got[key]["invariant_violations"]})

    with open(os.path.join(ROOT, a.out), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"reference: {REFERENCE}\n")
    print(f"{'order':<11}{'mode':<8}{'L':>4}{'k':>6}{'nv_recall':>11}{'lcs_word':>10}  identical")
    for r in rows:
        print(f"{r['order']:<11}{r['mode']:<8}{r['L']:>4}{r['k']:>6}"
              f"{r['nv_recall']:>11}{r['lcs_word']:>10}  {r['identical_to_reference']}")
    print(f"\n{len(rows)} arms compared, {disagreements} differ from the reference")
    print("-> CONFIRMED: the order does not reach an unconstrained decode"
          if disagreements == 0 else
          "-> REFUTED: the constraint reaches a decode that is supposed to be unconstrained. "
          "Do not quote Table 3 against a shared baseline until this is understood.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
