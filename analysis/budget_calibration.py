"""feat-168: pick the budget that binds as hard on a new corpus as the committed one does on ours.

A budget is not transferable between corpora; what transfers is how hard it binds. feat-166 proved
that the expensive way -- at k=10 on AlpacaEval the constraint is active on 0.016% of steps against
8.376% on our own prompt set, so 794 of 805 served completions were the unconstrained risky model
and the arm labelled "metered decoder" was not one.

The selection rule and the grid are fixed in results/onset_prediction_mixtral_power_k.md, committed
before the sweep ran. This script only applies it: argmin |activity(k) - target|, with G-cal (the
grid must BRACKET the target) checked first, because the nearest endpoint of a one-sided grid is a
ceiling rather than a match and caution (g) is what a ceiling does to an interval.
"""
import argparse
import csv
import glob
import json
import os

# output/phase2/conc_all, the committed metered arm, over the three counters that partition the
# true decoded length (caution (ah)): 261,239 active of 3,118,893.
TARGET = 261239 / 3118893


def activity(gen_dir):
    """(active steps, total decode steps, n trajectories) for one arm."""
    act = tot = n = 0
    for f in sorted(glob.glob(os.path.join(gen_dir, "trajectories_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            a = json.loads(line)["aggregate"]
            n += 1
            act += a.get("steps_active", 0)
            tot += (a.get("steps_active", 0) + a.get("steps_forced_safe", 0)
                    + a.get("steps_risky_unchanged", 0))
    assert n, gen_dir
    assert tot, f"{gen_dir}: no decode steps counted"
    return act, tot, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/mixpow")
    ap.add_argument("--grid", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--target", type=float, default=TARGET)
    ap.add_argument("--out", default="results/mixpow_kcal.csv")
    a = ap.parse_args()

    rows = []
    for k in a.grid:
        d = os.path.join(a.root, "kcal_" + str(k).replace(".", ""))
        act, tot, n = activity(d)
        rows.append({"k": k, "steps_active": act, "steps_total": tot, "n_trajectories": n,
                     "activity": round(act / tot, 8),
                     "abs_dist_to_target": round(abs(act / tot - a.target), 8)})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"target activity {a.target:.5f} (committed metered arm, output/phase2/conc_all)")
    print(f"{'k':>7}{'active':>10}{'total':>10}{'activity':>11}{'|dist|':>11}")
    for r in rows:
        print(f"{r['k']:>7}{r['steps_active']:>10}{r['steps_total']:>10}"
              f"{r['activity']:>11.5f}{r['abs_dist_to_target']:>11.5f}")

    above = [r for r in rows if r["activity"] > a.target]
    below = [r for r in rows if r["activity"] < a.target]
    if not above or not below:
        print("\nG-cal FAIL: the grid does not bracket the target "
              f"({len(above)} above, {len(below)} below). NOT RUN -- an endpoint is a ceiling, "
              "not a match, and the registration forbids taking the nearest one.")
        return
    print(f"\nG-cal PASS: {len(above)} above the target, {len(below)} below.")
    # argmin of the distance; ties go to the smaller k, which sorted() gives for free.
    best = min(sorted(rows, key=lambda r: r["k"]), key=lambda r: r["abs_dist_to_target"])
    print(f"CHOSEN k = {best['k']}  (activity {best['activity']:.5f}, "
          f"target {a.target:.5f}, ratio {best['activity'] / a.target:.2f}x)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
