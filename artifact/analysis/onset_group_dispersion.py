"""Dispersion of the nine-pair onset ratios, between pairs and within one pair re-seeded.

Appendix~\\ref{app:seed} reads a subgroup of the nine-pair table as tight. That reading is only
available if the tightness is larger than the noise of measuring one row, and until the seed ladders
existed the paper had no estimate of that noise at all: every one of the nine memorisers is a single
fine-tune at --seed 0.

Two of the nine rows have since been re-trained at three seeds each, on this same corpus, with the
anchor, passages, grid and recipe held fixed (analysis/strength_ladder.py, pairs kl3m_cb and
pleias_cb). Their corners reproduce the table's own entries, so the ladder re-measures the row
rather than measuring something adjacent to it. This script puts the between-pair dispersion and
the within-pair dispersion in one table so the appendix can be checked against it mechanically
rather than by eye -- caution (j), which put six of seventy-two cells one off in the last digit.

    .venv/bin/python analysis/onset_group_dispersion.py   ->  results/onset_group_dispersion.csv
"""
import argparse
import csv
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The grouping is the appendix's own, read off the seed words a fixed 20-token seed buys.
GT10 = ("TinyComma-1.8B", "Pleias-1.2B", "Comma-7B", "Phi-3.5-mini", "Pleias-350M")
LE10 = ("open-calm-1b", "open-calm-3b", "KL3M-520M", "KL3M-1.7B")

# The two ladders that re-trained a row of the table itself, on the table's own corpus.
LADDERS = {
    "reseed_pleias12b_copybench": ("results/strength_ladder_pleias_cb_seeds.csv", "Pleias-1.2B"),
    "reseed_kl3m520m_copybench": ("results/strength_ladder_kl3m_cb_seeds.csv", "KL3M-520M"),
}


def stats(label, values, member=""):
    v = sorted(values)
    return {
        "subset": label,
        "n": len(v),
        "member_of": member,
        "mean": round(st.mean(v), 6),
        "sd": round(st.stdev(v), 6),
        "cv_pct": round(100 * st.stdev(v) / st.mean(v), 4),
        "span": round(v[-1] - v[0], 6),
        "lo": round(v[0], 6),
        "hi": round(v[-1], 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/onset_group_dispersion.csv")
    args = ap.parse_args()

    table = {}
    with open(os.path.join(ROOT, "results/onset_table.csv")) as fh:
        for r in csv.DictReader(fh):
            if " + " in r["pair"]:
                table[r["pair"].split(" + ")[0]] = float(r["ratio"])
    missing = [p for p in GT10 + LE10 if p not in table]
    assert not missing, f"onset_table.csv is missing {missing}"

    rows = [
        stats("all_nine", table.values()),
        stats("adversary_holds_gt10_words", [table[p] for p in GT10]),
        stats("adversary_holds_le10_words", [table[p] for p in LE10]),
    ]
    # The separation between the two families, which is what the units claim rests on.
    gt, le = st.mean([table[p] for p in GT10]), st.mean([table[p] for p in LE10])
    rows.append({"subset": "family_mean_gap", "n": 9, "member_of": "", "mean": round(le - gt, 6),
                 "sd": "", "cv_pct": "", "span": round(le - gt, 6),
                 "lo": round(gt, 6), "hi": round(le, 6)})

    for label, (path, member) in LADDERS.items():
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            raise SystemExit(f"{path} missing; run analysis/strength_ladder.py first")
        with open(full) as fh:
            lad = [float(r["ratio"]) for r in csv.DictReader(fh) if r.get("ratio")]
        assert len(lad) >= 3, f"{path} has {len(lad)} points, need at least 3"
        # The ladder's corner IS the table's row; if it drifts, these are not the same measurement.
        assert min(abs(x - table[member]) for x in lad) < 5e-4, (
            f"{label}: no ladder point reproduces {member}'s table entry {table[member]}")
        rows.append(stats(label, lad, member))

    out = os.path.join(ROOT, args.out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(f"{r['subset']:<28} n={r['n']:<2} mean {r['mean']:<9} sd {str(r['sd']):<9} "
              f"cv {str(r['cv_pct']):<8} span {r['span']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
