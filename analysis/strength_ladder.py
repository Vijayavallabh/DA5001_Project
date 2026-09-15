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

import argparse, csv, itertools, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset import crossing, curve            # noqa: E402
from analysis.seed_effect import spearman             # noqa: E402

# (label, sweep dir). The 40-epoch point is feat-120's own run, already measured and bootstrapped;
# it is the SAME protocol, so it belongs on the ladder rather than beside it. Its sweep directory is
# the merged one, because its committed grid needed the licensed extension (caution (g)).
POINTS = [
    ("epochs=10", "output/phase5/fineb_pleias_e10"),
    ("epochs=20", "output/phase5/fineb_pleias_e20"),
    ("epochs=30", "output/phase5/fineb_pleias_e30"),
    ("epochs=40 (feat-120)", "output/phase5/fineb_pleias12b_full"),
]
S_X = 3.048079572669047      # Pleias-1.2B on BookMIA, results/onset_theory_bookmia.csv
ENTRY_GATE = 0.10            # AGENTS.md caution (a): SAMPLED, never greedy
# committed in results/onset_prediction_strength.md before any of these memorisers existed
RHO_EXPLAINS, SPAN_EXPLAINS = -0.8, 0.15
SPAN_REFUTES, STRENGTH_SPAN_MIN = 0.10, 3.0


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
    ap.add_argument("--thresh", type=float, default=0.01)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    for label, run in POINTS:
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
            s_x=round(S_X, 4), onset_lo=lo, onset_hi=hi,
            onset=round(est, 4) if est else None,
            ratio=round(est / S_X, 4) if est else None,
            grid=" ".join(f"{k:g}:{v:.3f}" for k, v in c.items())))

    if not rows:
        print("[sl] nothing to score", file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "strength_ladder.csv")
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
