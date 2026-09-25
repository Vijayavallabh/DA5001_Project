"""Review 3 (seventh round, Q4): does the difference of GAINS agree with the direct difference of LEVELS?

Table 2's last column is the direct difference, selection's order-averaged level minus the meter's, both
against one opponent in one pass. The headline quotes D3 instead: selection's gain over its own n=1
control minus the meter's gain over its own anchor, which removes whatever the two pipelines' zero-budget
arms disagree on. Both are read here from the committed per-prompt levels of each row's own pass, and each
is labelled by the side of zero its 95% paired bootstrap interval lies on. No GPU, no judge call.

Usage: .venv/bin/python analysis/d3_vs_direct.py --out results
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import paired_boot  # noqa: E402

FL = ("frontier_levels_per_prompt_a", "frontier_levels_per_prompt_b", "frontier_levels_per_prompt_c")
OA = "order_averaged_h2h_per_prompt_"
# Table 2 row -> (per-prompt files of its pass, meter column); selection u_sel_n64, controls u_sel_n1 and
# u_anchor_k0 of the same pass.
ROWS = [
    ("8B text, T=1.0", 0.5, FL, "u_met_k0.5"), ("8B text, T=1.0", 1, FL, "u_met_k1"),
    ("8B text, T=1.0", 10, FL, "u_met_k10"),
    ("70B base, T=1.0", 0.5, (OA + "he70b_k05",), "u_metered_k0.5"),
    ("70B base, T=1.0", 1, (OA + "he70b_k20",), "u_met70b_k1"),
    ("70B base, T=1.0", 20, (OA + "he70b_k20",), "u_metered_k20"),
    ("8B text, T=0.7", 0.5, (OA + "t07_8b_k10",), "u_met8b_k0.5"),
    ("8B text, T=0.7", 1, (OA + "t07_8b_k1",), "u_metered_k1"),
    ("8B text, T=0.7", 10, (OA + "t07_8b_k10",), "u_metered_k10"),
    ("70B base, T=0.7", 0.5, (OA + "t07_70b_k20",), "u_met70b_k0.5"),
    ("70B base, T=0.7", 1, (OA + "t07_70b_k1",), "u_metered_k1"),
    ("70B base, T=0.7", 20, (OA + "t07_70b_k20",), "u_metered_k20"),
    ("AnchoredByte", 0.1, (OA + "ab70_k0.1",), "u_metered_k0.1"),
    ("AnchoredByte", 0.5, (OA + "ab70_k0.5",), "u_metered_k0.5"),
    ("AnchoredByte", 2, (OA + "ab70_k2",), "u_metered_k2"),
    ("8B chat template", 0.5, (OA + "chatgrid_k05",), "u_metered_k0.5"),
    ("8B chat template", 1, (OA + "served_k1chat",), "u_metered_k1"),
    ("8B chat template", 2, (OA + "chatgrid_k05",), "u_chat_k2"),
    ("8B chat template", 3, (OA + "chatgrid_k3",), "u_metered_k3"),
    ("8B chat template", 5, (OA + "chatgrid_k3",), "u_chat_k5"),
    ("8B chat template", 10, (OA + "served_k10chat",), "u_metered_k10"),
]


def label(lo, hi):
    return "UNRESOLVED" if lo <= 0 <= hi else ("CONFIRMED" if lo > 0 else "REFUTED")


def load(results, files):
    per = {}
    for f in files:
        for r in csv.DictReader(open(os.path.join(results, f + ".csv"))):
            per.setdefault(r["prompt_id"], {}).update({k: float(v) for k, v in r.items() if k.startswith("u_")})
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=7717)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = []
    for block, k, files, met in ROWS:
        per = load(a.results, files)
        pids = sorted(per)
        direct = [per[p]["u_sel_n64"] - per[p][met] for p in pids]
        d3 = [(per[p]["u_sel_n64"] - per[p]["u_sel_n1"]) - (per[p][met] - per[p]["u_anchor_k0"]) for p in pids]
        lo, hi = paired_boot(direct, rng)
        lo3, hi3 = paired_boot(d3, rng)
        row = dict(block=block, k=k, source=files[0].replace(OA, ""), n=len(pids),
                   direct=round(sum(direct) / len(pids), 4), direct_lo95=round(lo, 4), direct_hi95=round(hi, 4),
                   direct_reading=label(lo, hi), d3=round(sum(d3) / len(pids), 4), d3_lo95=round(lo3, 4),
                   d3_hi95=round(hi3, 4), d3_reading=label(lo3, hi3),
                   controls_differ=round(sum(per[p]["u_sel_n1"] - per[p]["u_anchor_k0"] for p in pids) / len(pids), 4))
        row["same_reading"] = row["direct_reading"] == row["d3_reading"]
        out.append(row)
        print(f"{block:18s} k={k:<4} direct {row['direct']:+.4f} {row['direct_reading']:10s} "
              f"D3 {row['d3']:+.4f} [{row['d3_lo95']:+.4f}, {row['d3_hi95']:+.4f}] {row['d3_reading']:10s} "
              f"controls {row['controls_differ']:+.4f}")
    path = os.path.join(a.out, "d3_vs_direct.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"same reading on {sum(r['same_reading'] for r in out)} of {len(out)} rows; wrote {path}")


if __name__ == "__main__":
    main()
