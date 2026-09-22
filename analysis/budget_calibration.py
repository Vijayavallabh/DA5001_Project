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
import sys

# `output/phase2/conc_all` is a k-SWEEP over six budgets and six prompt classes, not an arm. The
# first version of this file set TARGET = 261239/3118893 = 0.08376 by scanning the whole directory,
# which pools every k (including k=0.5, which binds on 48% of steps) and every class (including the
# protected corpus). The arm the paper judges is k=10 over the three ordinary classes alone, and it
# reads 0.000080 -- the typed constant was wrong by a factor of 1046, and a false comparison built
# on it reached the compiled manuscript.
#
# That is caution (v) exactly: a reference number carries its protocol. So the target is no longer
# typed at all -- it is computed from the arm the judge actually loads, by the same (k, classes)
# selection `analysis/selection_decoding.py` uses, and written to the CSV beside the sweep it gates.
REF_DIR = "output/phase2/conc_all"
REF_K = "10"


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def committed_activity(ref_dir=REF_DIR, k=REF_K):
    """The committed metered arm's own activity, filtered exactly as the judge filters it."""
    from analysis.utility import CLASSES
    act = tot = n = 0
    for cls in CLASSES:
        path = os.path.join(ref_dir, f"trajectories_k{k}_{cls}.jsonl")
        assert os.path.exists(path), f"{path} missing; the reference cannot be derived"
        for line in open(path, encoding="utf-8"):
            a = json.loads(line)["aggregate"]
            n += 1
            act += a.get("steps_active", 0)
            tot += (a.get("steps_active", 0) + a.get("steps_forced_safe", 0)
                    + a.get("steps_risky_unchanged", 0))
    assert tot, ref_dir
    return act / tot, act, tot, n


def activity(gen_dir, k=None):
    """(active steps, total decode steps, n trajectories) for one arm at one budget.

    A DIRECTORY IS NOT A BUDGET. output/phase2/conc_all holds k in {0.5, 1, 3, 5, 10, 20} and
    summing it gives the mean over the whole sweep: that is how this arm's reference activity was
    once read as 8.376% instead of 0.008%, a factor of 1046, and the wrong number reached the
    compiled PDF. Pass `k` to select one budget; with k=None the directory must hold exactly one,
    and the assertion below is what stops a sweep being summed by accident a second time."""
    pat = ("trajectories_*.jsonl" if k is None
           else f"trajectories_k{k:g}_*.jsonl".replace("k1_", "k1_"))
    files = sorted(glob.glob(os.path.join(gen_dir, pat)))
    if k is None:
        ks = {os.path.basename(f).split("_")[1] for f in files}
        assert len(ks) <= 1, (
            f"{gen_dir} holds {len(ks)} budgets {sorted(ks)}; pass k= to pick one rather than "
            "summing a sweep")
    assert files, f"{gen_dir}: no trajectories for k={k}"
    act = tot = n = 0
    for f in files:
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
    ap.add_argument("--target", type=float, default=0.0,
                    help="0 (default) derives it from the committed arm; never type it in")
    ap.add_argument("--out", default="",
                    help="default: results/<basename of --root>_kcal.csv, so two workloads can "
                         "never share one file")
    a = ap.parse_args()
    # CAUTION (ax): TWO PASSES MUST NOT SHARE A FILENAME. The default used to be the literal
    # results/mixpow_kcal.csv whatever --root was, so an MT-Bench calibration and an AlpacaEval
    # one wrote the same file. MT-Bench's grid was then never committed anywhere, and feat-173's
    # scored G-cal row was filled in from AlpacaEval's numbers -- 0.765/0.684/0.077/0.005/0.0002
    # where MT-Bench's own are 0.659/0.589/0.073/0.006/0.0003. The choice of k was unaffected
    # (the argmin is k=1.0 under either grid) but the printed evidence was another arm's.
    if not a.out:
        a.out = os.path.join("results", os.path.basename(a.root.rstrip("/")) + "_kcal.csv")

    if not a.target:
        a.target, racr, rtot, rn = committed_activity()
        print(f"reference derived from {REF_DIR} k={REF_K} over the judge's own classes: "
              f"{racr}/{rtot} = {a.target:.6f}  (n={rn})")
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
