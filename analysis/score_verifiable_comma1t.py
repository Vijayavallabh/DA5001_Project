"""Score results/onset_prediction_verifiable_comma1t.md (feat-137) against its committed bands.

Zero GPU. Written BEFORE the arm produced a number and mutation-tested before the data existed
(caution (v)), so every reading rule here is the pre-registration's.

The question: the judge-free axis is the one that exists to answer every objection to the judge, and
it rests on ONE anchor. This adds a second, chosen so it answers an open question rather than merely
adding a row -- common-pile/comma-v0.1-1t is the same architecture at the same 7B scale as Comma-7B
and differs only in corpus size, so feat-136's comma1thb and this arm are the judged and judge-free
halves of one training-data ablation.

The gates, in the order they are applied:

  G0  the FLOOR gate, and the reason this arm can produce a null that means anything. An anchor that
      cannot do the task at all produces a flat curve, and a flat curve from a floor is NOT a
      saturation. n=1 majority-vote accuracy must be >= 0.05 (Comma-7B reads 0.320). Below it the arm
      is UNINFORMATIVE, not SATURATED, and the band is NOT computed.
  G1  coverage: n in {1,2,4,8,16,32,64} on all 500 problems under both rules. BLOCKS the band.

The band is the gain at n=64 under MAJORITY VOTE, which selection_verifiable.py already computes as a
paired bootstrap of acc(arm) - acc(n=1) over the same resampled problems. Majority vote is the
headline because it has no scorer to overoptimise against: on record it gains +0.222 against the
pointwise reward's +0.066, and caution (ao) records the pointwise rule's accuracy FALLING with n on
TriviaQA.

Usage: .venv/bin/python analysis/score_verifiable_comma1t.py --out results
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NAME = "comma1t"
LABEL = "Comma-7B (1T tokens)"
MAJORITY = "majority vote (self-consistency)"
POINTWISE = "pointwise reward (Qwen2.5-7B)"
GRID = (1, 2, 4, 8, 16, 32, 64)
N_PROBLEMS = 500
FLOOR = 0.05                      # G0
# The arm on record, for the secondary ordering check only. Never for a level comparison: the
# pre-registration excludes comparing accuracy LEVELS across anchors as though a difference were a
# finding, because they are different models.
ON_RECORD_MAJORITY_GAIN = 0.222
ON_RECORD_POINTWISE_GAIN = 0.066


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else None


def g0_floor(rs):
    """The floor gate. Returns (ok, message, acc1)."""
    r = [x for x in rs if x["arm"] == MAJORITY and int(float(x["n"])) == 1]
    if not r:
        return False, "no n=1 majority-vote row", float("nan")
    acc = float(r[0]["acc"])
    return acc >= FLOOR, (f"n=1 majority-vote accuracy {acc:.4f} "
                          f"(>= {FLOOR} required; Comma-7B reads 0.320)"), acc


def g1_coverage(rs):
    for rule in (MAJORITY, POINTWISE):
        sub = [x for x in rs if x["arm"] == rule]
        if not sub:
            return False, f"no rows for {rule!r}; arms present: {sorted({x['arm'] for x in rs})}"
        have = sorted({int(float(x["n"])) for x in sub})
        if have != sorted(GRID):
            return False, f"{rule}: grid is {have}, not {sorted(GRID)}"
        counts = {int(float(x["n_problems"])) for x in sub}
        if counts != {N_PROBLEMS}:
            return False, f"{rule}: n_problems is {counts}, not {{{N_PROBLEMS}}}"
    return True, f"both rules on {sorted(GRID)} over {N_PROBLEMS} problems"


def band(rs, rule=MAJORITY, n=64):
    r = next(x for x in rs if x["arm"] == rule and int(float(x["n"])) == n)
    return float(r["gain"]), float(r["gain_lo95"]), float(r["gain_hi95"]), float(r["acc"])


def verdict_for(g, lo, hi):
    if lo > 0:
        return "CLIMBS"
    if hi < 0:
        return "TURNS OVER"
    return "NO CLIMB"


CROSS = {
    ("CLIMBS", "CLIMBS"): "Corpus size gates NEITHER axis; the 7B result is about architecture and "
                          "scale.",
    ("SATURATED BY 8", "NO CLIMB"): "The strongest H3 reading: corpus size gates the mechanism at "
                                    "fixed size, on a task with no judge in it, so the finding "
                                    "cannot be blamed on the judge.",
    ("SATURATED BY 8", "CLIMBS"): "The judged null at this anchor is an artefact of the judge or "
                                  "the scorer, not a limit of the anchor -- and that weakens every "
                                  "judged null this paper reports.",
    ("CLIMBS", "NO CLIMB"): "THE MOST DAMAGING READING: a judged gain at this anchor that no "
                            "verifiable improvement backs. It does not refute the certificate, "
                            "which is a bound and not a utility claim, but it puts a named limit "
                            "on what the judged gains mean.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = a.out
    rs = rows(os.path.join(out, f"selection_verifiable_{NAME}.csv"))

    print(f"results/onset_prediction_verifiable_{NAME}.md (feat-137) -- does the judge-free climb "
          f"survive a change of anchor?")
    print(f"  on record at Comma-7B: majority vote {ON_RECORD_MAJORITY_GAIN:+.3f} at n=64, "
          f"pointwise reward {ON_RECORD_POINTWISE_GAIN:+.3f}. No judge is involved anywhere: the "
          f"metric is exact match.")

    if rs is None:
        print(f"\n  G0/G1: the arm has not finished; {out}/selection_verifiable_{NAME}.csv is absent")
        print("  NOT SCORED.")
        return 1

    ok0, msg0, acc1 = g0_floor(rs)
    print(f"\n  G0 floor:    {'PASS' if ok0 else 'FAIL'} -- {msg0}")
    if not ok0:
        print("  UNINFORMATIVE, not SATURATED: an anchor that cannot do the task produces a flat")
        print("  curve, and a flat curve from a floor is not a saturation. The band is NOT computed.")
        return 1

    ok1, msg1 = g1_coverage(rs)
    print(f"  G1 coverage: {'PASS' if ok1 else 'FAIL'} -- {msg1}")
    if not ok1:
        print("  NOT SCORED: the band is not computed.")
        return 1

    g, lo, hi, acc64 = band(rs, MAJORITY, 64)
    v = verdict_for(g, lo, hi)
    print(f"\n  MAJORITY VOTE, gain at n=64 (paired over the same 500 problems): "
          f"{g:+.4f} [{lo:+.4f}, {hi:+.4f}]  -> {v}")
    print(f"    accuracy {acc1:.4f} at n=1 -> {acc64:.4f} at n=64")

    pg, plo, phi, pacc = band(rs, POINTWISE, 64)
    pv = verdict_for(pg, plo, phi)
    print(f"  pointwise reward, gain at n=64: {pg:+.4f} [{plo:+.4f}, {phi:+.4f}]  -> {pv}")
    ordering_holds = g > pg
    print(f"  majority vote beats the pointwise reward: {'YES' if ordering_holds else 'NO'} "
          f"({g:+.4f} against {pg:+.4f}; on record {ON_RECORD_MAJORITY_GAIN:+.3f} against "
          f"{ON_RECORD_POINTWISE_GAIN:+.3f})")
    if not ordering_holds:
        print("    THE ORDERING INVERTS. The appendix's reason for making majority vote the")
        print("    judge-free headline is anchor-specific and must be scoped to the anchor it was")
        print("    measured on (caution (ao)).")

    for r in sorted(rs, key=lambda x: (x["arm"], int(float(x["n"] or 1)))):
        sp = r.get("spearman_acc_logn", "")
        print(f"    {r['arm'][:34]:<34} n={int(float(r['n'] or 1)):<3} acc={float(r['acc']):.4f} "
              f"gain={r['gain'] or '--':<8} spearman={sp}")

    # ---- the cross-axis reading, decided by a table committed before either arm ran -------------
    print("\n=== the cross-axis reading (committed in advance, all four cases) ===")
    bl = rows(os.path.join(out, "breadth_ladders_scoring.csv"))
    judged = None
    if bl:
        row = next((x for x in bl if x["name"] == "comma1thb"), None)
        judged = row["verdict"] if row else None
    if judged in (None, "", "NOT SCORED"):
        print("  feat-136's comma1thb is not scored yet, so the ablation is open on the judged half.")
        cross = ""
    else:
        cross = CROSS.get((judged, v), f"judged {judged} against judge-free {v}: not one of the four "
                                       f"cases the pre-registration enumerated; report it as it is "
                                       f"and do not force it into one.")
        print(f"  judged (comma1thb) {judged}  x  judge-free {v}")
        print(f"  -> {cross}")

    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"verifiable_{NAME}_scoring.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["anchor", "rule", "acc_n1", "acc_n64", "gain64", "lo95", "hi95", "verdict",
                    "beats_pointwise", "judged_verdict", "cross_axis_reading"])
        w.writerow([LABEL, "majority vote", round(acc1, 4), round(acc64, 4), round(g, 4),
                    round(lo, 4), round(hi, 4), v, "yes" if ordering_holds else "no",
                    judged or "", cross])
        w.writerow([LABEL, "pointwise reward", round(acc1, 4), round(pacc, 4), round(pg, 4),
                    round(plo, 4), round(phi, 4), pv, "", judged or "", ""])
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
