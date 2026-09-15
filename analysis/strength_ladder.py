"""Is the onset ratio a property of the corpus, or of the memoriser? (the corpus-vs-memoriser arm)

feat-120 changed the protected work a second time and the onset ordering inverted: Pleias-1.2B rose
from 0.878 and 0.895 on the first two corpora to 1.3142 on BookMIA, and the paper retracted the
claim that leakage beginning after vacuity is a property of the pair. That file recorded, in
advance, the rival explanation it could not settle -- a weak memoriser needs more budget before it
can leak -- and could not settle it because every memoriser is fine-tuned on the corpus it is then
measured against, so corpus and strength move together by construction.

This holds the pair and the corpus fixed (Pleias-1.2B on BookMIA, the cell that inverted) and varies
--epochs alone. The anchor never changes, so s(x) is the SAME denominator at every point and the
ratios are directly comparable.

Strength is MEASURED, by each point's own sampled k=-1 arm, never by the epoch count: the knob does
not have to be monotone and no band assumes it is. Bands are committed in
results/onset_prediction_strength.md, before any of these memorisers existed.

  .venv/bin/python analysis/strength_ladder.py --out results

One convention, stated because two exist. The onset here comes from analysis.onset.crossing, which
interpolates the SUMMARY curve; analysis/onset_ci.py recomputes from the per-passage CSV and reports
the bootstrap interval, and on the epochs=40 point the two read 4.0071 and 4.0058 -- a difference of
0.0013 nats, 0.03%, far below the precision of any claim. All four points here go through the same
call, so the ladder is internally consistent; the manuscript quotes onset_ci's value wherever it
quotes an interval beside it, because the point and the interval must come from one computation.

Writes <out>/strength_ladder.csv. No GPU.
"""
from __future__ import annotations

import argparse, csv, itertools, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset import crossing, curve            # noqa: E402
from analysis.seed_effect import spearman             # noqa: E402

# (label, sweep dir). The 40-epoch point is feat-120's own run, already measured and bootstrapped;
# it is the SAME protocol, so it belongs on the ladder rather than beside it. Its sweep directory is
# the merged one, because its committed grid needed the licensed extension (caution (g)).
CORNER = ("output/phase5/fineb_pleias12b_full", "epochs=40, seed=0 (feat-120)")
AXES = {
    # feat-121: vary --epochs at seed 0
    "epochs": [("epochs=10", "output/phase5/fineb_pleias_e10"),
               ("epochs=20", "output/phase5/fineb_pleias_e20"),
               ("epochs=30", "output/phase5/fineb_pleias_e30"),
               ("epochs=40 (feat-120)", CORNER[0])],
    # the seed arm: vary --seed at 40 epochs. Both axes end on the SAME corner run, which is what
    # makes their two spans comparable rather than two unrelated numbers.
    "seeds":  [("seed=1", "output/phase5/fineb_pleias_s1"),
               ("seed=2", "output/phase5/fineb_pleias_s2"),
               ("seed=3", "output/phase5/fineb_pleias_s3"),
               ("seed=0 (feat-120)", CORNER[0])],
}
POINTS = AXES["epochs"]      # feat-121's default path, unchanged
S_X = 3.048079572669047      # Pleias-1.2B on BookMIA, results/onset_theory_bookmia.csv
# A second pair's seed ladder (the strongest BookMIA memoriser and the fine-tokenizer pair), on ITS
# OWN corner grid -- no licensed extension, because its no-crossing there was 0.0%.
KL3M_S_X = 2.4316326727999997
KL3M_CORNER = "output/phase5/fineb_kl3m520m"
# The same anchor on CopyBench -- the pair that is IN the nine-pair table of Section 4, so this one
# tests the claim directly instead of by analogy. Its s(x) and grid are the table's own, from
# results/onset_ci.csv; the corner is the run the table quotes.
CB_S_X = 2.4147499999999997
CB_CORNER = "output/phase5/fine_kl3m520m"
PAIRS = {
    "pleias": dict(s_x=S_X, axes=None),          # axes filled in below from AXES
    "kl3m": dict(s_x=KL3M_S_X, axes={"seeds": [
        ("seed=1", "output/phase5/fineb_kl3m_s1"),
        ("seed=2", "output/phase5/fineb_kl3m_s2"),
        # seeds 3 and 4 were added at 03:25 under the amendment recorded in
        # results/onset_prediction_seedspread2.md, before any result of the arm existed. They are
        # barred from the COMMITTED primary, which is the {0,1,2} span, because a five-point span
        # is not a three-point span and Pleias has three.
        ("seed=3", "output/phase5/fineb_kl3m_s3"),
        ("seed=4", "output/phase5/fineb_kl3m_s4"),
        ("seed=0 (feat-120)", KL3M_CORNER)]}),
    "kl3m_cb": dict(s_x=CB_S_X, axes={"seeds": [
        ("seed=1", "output/phase5/finec_kl3m520m_s1"),
        ("seed=2", "output/phase5/finec_kl3m520m_s2"),
        ("seed=0 (the table's own)", CB_CORNER)]}),
}
PAIRS["pleias"]["axes"] = AXES
ENTRY_GATE = 0.10            # AGENTS.md caution (a): SAMPLED, never greedy
# committed in results/onset_prediction_strength.md before any of these memorisers existed
RHO_EXPLAINS, SPAN_EXPLAINS = -0.8, 0.15
SPAN_REFUTES, STRENGTH_SPAN_MIN = 0.10, 3.0
# committed in results/onset_prediction_seedspread.md before any seed memoriser existed, and read
# against feat-121's MEASURED epoch-only span of 0.4721 on the identical pair, corpus and grid
SEED_SPAN_NOISE, SEED_SPAN_STRENGTH, EPOCH_SPAN_MEASURED = 0.20, 0.10, 0.4721
# results/onset_prediction_seedspread2.md fixes the cross-pair comparison at these three seeds for
# BOTH pairs, because max-minus-min grows with the number of draws and the pairs have different n.
# The rule is about seed NUMBERS, not label text: every pair's committed primary is seeds 0, 1, 2,
# which is the set all three pairs have, so no span is ever read against a span of different n.
PRIMARY_SEEDS = (0, 1, 2)


def seed_of(label):
    """The seed a point is, or None if the point is not on a seed axis."""
    m = re.match(r"seed=(\d+)", label)
    return int(m.group(1)) if m else None


def baseline(path, k):
    for r in csv.DictReader(open(path)):
        if r["mode"] == "single" and r["L"] == "0" and float(r["k"]) == k:
            return float(r["nv_recall_mean"])
    return None


def exact_p(a, b):
    """Two-sided permutation p over all orderings of b. n = 4 here, so the floor is 1/12."""
    obs = abs(spearman(a, b))
    perms = list(itertools.permutations(b))
    return sum(1 for q in perms if abs(spearman(a, list(q))) >= obs - 1e-12) / len(perms)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="pleias", choices=sorted(PAIRS),
                    help="pleias carries both ladders; kl3m carries the seed ladder only")
    ap.add_argument("--axis", default="epochs", choices=sorted(AXES) + ["pooled"],
                    help="epochs = feat-121; seeds = the seed arm; pooled = the committed "
                         "secondary over both ladders' seven distinct points")
    ap.add_argument("--thresh", type=float, default=0.01)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    s_x = PAIRS[a.pair]["s_x"]
    axes = PAIRS[a.pair]["axes"]
    if a.axis not in axes and a.axis != "pooled":
        print(f"[sl] pair {a.pair} has no {a.axis} ladder", file=sys.stderr)
        return 1
    if a.axis == "pooled":
        seen, points = set(), []
        for ax in sorted(axes):
            for lab, run in axes[ax]:
                if run not in seen:
                    seen.add(run); points.append((lab, run))
    else:
        points = axes[a.axis]

    rows = []
    for label, run in points:
        path = os.path.join(run, "composition_summary.csv")
        if not os.path.exists(path):
            print(f"[sl] no sweep at {path}, skipping {label}", file=sys.stderr)
            continue
        c = curve(path, "single", 0)
        lo, hi, est = crossing(c, a.thresh)
        gate = baseline(path, -1.0)
        rows.append(dict(
            point=label, run=run, n_grid=len(c),
            k_minus1_recall=gate, k_zero_recall=baseline(path, 0.0),
            entered=(gate is not None and gate >= ENTRY_GATE),
            s_x=round(s_x, 4), onset_lo=lo, onset_hi=hi,
            onset=round(est, 4) if est else None,
            ratio=round(est / s_x, 4) if est else None,
            grid=" ".join(f"{k:g}:{v:.3f}" for k, v in c.items())))

    if not rows:
        print("[sl] nothing to score", file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    stem = a.axis if a.pair == "pleias" else f"{a.pair}_{a.axis}"
    name = "strength_ladder.csv" if (a.pair == "pleias" and a.axis == "epochs") \
        else f"strength_ladder_{stem}.csv"
    path = os.path.join(a.out, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("One pair, one corpus, one knob. Bands committed in")
    print("results/onset_prediction_strength.md before any of these memorisers existed.\n")
    print(f"{'point':24s}{'sampled k=-1':>14s}{'k=0':>7s}{'onset':>9s}{'bracket':>14s}{'ratio':>9s}")
    for r in rows:
        br = f"({r['onset_lo']},{r['onset_hi']}]" if r["onset_lo"] else "--"
        o = f"{r['onset']:.3f}" if r["onset"] else "none"
        ra = f"{r['ratio']:.4f}" if r["ratio"] else "--"
        print(f"{r['point']:24s}{r['k_minus1_recall']:>14.4f}{r['k_zero_recall']:>7.3f}"
              f"{o:>9s}{br:>14s}{ra:>9s}")

    ok = [r for r in rows if r["entered"] and r["ratio"]]
    excluded = [r for r in rows if not r["entered"]]
    for r in excluded:
        print(f"\n  EXCLUDED (sampled k=-1 {r['k_minus1_recall']:.4f} < {ENTRY_GATE}): {r['point']}"
              "  -- a statement about that memoriser, not about the question")
    if len(ok) < 3:
        print(f"\n  only {len(ok)} points entered; the ladder cannot be scored")
        return 0

    strengths = [r["k_minus1_recall"] for r in ok]
    ratios = [r["ratio"] for r in ok]
    span_strength = max(strengths) / min(strengths) if min(strengths) > 0 else float("inf")
    span_ratio = max(ratios) - min(ratios)
    rho, p = spearman(strengths, ratios), exact_p(strengths, ratios)

    print(f"\n  measured strength span {span_strength:.2f}x  ({min(strengths):.4f} to "
          f"{max(strengths):.4f})")
    print(f"  ratio span             {span_ratio:.4f}  ({min(ratios):.4f} to {max(ratios):.4f})")
    print(f"  rho(sampled k=-1, onset ratio) = {rho:+.3f}, exact p = {p:.4f} "
          f"(floor at n={len(ok)} is {1/len(list(itertools.permutations(range(len(ok))))) * 2:.3f})")

    if a.axis == "seeds":
        prim = [r for r in ok if seed_of(r["point"]) in PRIMARY_SEEDS]
        if len(prim) == len(PRIMARY_SEEDS):
            pv = [r["ratio"] for r in prim]
            print(f"\n  PRIMARY, the committed like-for-like span over {[r['point'] for r in prim]}:"
                  f" {max(pv) - min(pv):.4f}  ({min(pv):.4f} to {max(pv):.4f})")
            if len(ok) > len(prim):
                print(f"  secondary, all {len(ok)} points: {span_ratio:.4f}  -- reported beside the "
                      "primary and never in place of it, because a span grows with n")
        else:
            print(f"\n  the committed primary needs three of {list(PRIMARY_SEEDS)}; only "
                  f"{[r['point'] for r in prim]} entered")
        print(f"\n  read against feat-121's epoch-only span of {EPOCH_SPAN_MEASURED} on the same cell")
        if span_ratio >= SEED_SPAN_NOISE:
            verdict = (f"RUN-TO-RUN VARIATION: changing only the seed moves the ratio by "
                       f"{span_ratio:.4f}, at or above the committed {SEED_SPAN_NOISE}. The onset "
                       "ratio is not reproducible under this paper's own recipe, and the nine-pair "
                       "table's 0.2874 of between-pair structure sits inside the noise of a single "
                       "pair re-trained")
        elif span_ratio < SEED_SPAN_STRENGTH:
            verdict = (f"STRENGTH, NOT NOISE: the recipe reproduces at a fixed epoch count "
                       f"({span_ratio:.4f} < {SEED_SPAN_STRENGTH}), so feat-121's "
                       f"{EPOCH_SPAN_MEASURED} belongs to what the epochs changed, and the "
                       "nine-pair table is confounded by a variable it never controlled")
        else:
            verdict = "INCONCLUSIVE, and reported as inconclusive. No second seed set, no fifth seed"
        print(f"\n  -> {verdict}")
        print(f"wrote {path}")
        return 0

    if a.axis == "pooled":
        print(f"\n  pooled over both ladders: {len(ok)} distinct points on one pair and one corpus")
        print(f"  -> rho = {rho:+.3f}, exact p = {p:.4f}. Committed in advance as the secondary; "
              "reported whatever it says")
        print(f"wrote {path}")
        return 0

    if span_strength < STRENGTH_SPAN_MIN:
        verdict = (f"UNINFORMATIVE BY CONSTRUCTION: the knob produced a {span_strength:.2f}x "
                   f"strength span, below the committed {STRENGTH_SPAN_MIN}x. An arm that cannot "
                   "separate its levels cannot answer its question, whatever rho reads")
    elif rho <= RHO_EXPLAINS and span_ratio >= SPAN_EXPLAINS:
        verdict = ("STRENGTH EXPLAINS IT: within one pair and one corpus the onset ratio is a "
                   "function of memoriser strength. Per the pre-registration this does NOT restore "
                   "the retracted sentence -- it means the nine-pair table, whose memorisers were "
                   "never strength-matched, inherits the confound")
    elif span_ratio < SPAN_REFUTES and span_strength >= STRENGTH_SPAN_MIN:
        verdict = ("STRENGTH DOES NOT EXPLAIN IT: the ratio barely moves across a real strength "
                   "ladder, so feat-120's retraction stands unqualified and the corpus is doing "
                   "the work")
    else:
        verdict = "INCONCLUSIVE, and reported as inconclusive. No second ladder, no fifth point"
    print(f"\n  -> {verdict}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
