"""Score results/onset_prediction_within_host_spread.md (feat-141) against its committed bands.

Zero GPU. Written BEFORE any of its arms produced a number.

THE POINT OF THIS SCORER IS TO BE ABLE TO REFUTE feat-140. That arm concluded "a host change is not
a re-draw" from cross-host moves of 0.018-0.070 set against within-host moves of 0.007-0.018 -- but
the within-host figure came from four seed pairs, NONE at an anchor that flipped, and the three that
flipped had one host-B draw each. If those anchors are simply noisy, 0.0700 is an ordinary re-draw
for Pleias-3B and the conclusion is wrong. P1 names the number that withdraws it.

Constants are imported, never retyped (caution (j) one level up).
"""
import argparse
import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.score_breadth_ladders import (GRID, HALF_WIDTHS_FOR_STABLE, JUDGE_B,  # noqa: E402
                                            MAX_SEED_MOVE, N_PROMPTS, band, rows)
from analysis.score_seed_and_ladder import i1_length, i3_coverage, stat_row, verdict_of  # noqa: E402

# The cross-host move measured by feat-140, per anchor. These are ON RECORD and are not recomputed
# here; this arm asks only whether each is large relative to that anchor's OWN within-host spread.
CROSS_HOST = {"pleias12bhb": 0.0200, "pleias3bhb": 0.0700, "kl3m17bhb": 0.0180}

# P1's refutation threshold, fixed in the pre-registration: a within-host pair at Pleias-3B reaching
# this withdraws feat-140's conclusion. It is that anchor's own cross-host move, not a new constant.
REFUTES_AT = 0.0700

# Every host-B draw of each anchor, by the tag selection_scaling_<tag>.csv carries.
# The first draw of each was produced by run_breadth64.sh at its default seeds 42 43 44; the rest by
# run_breadth64_seed.sh. comma1t's first draw is tagged comma1thb64 because it ran under that name.
DRAWS = {
    "pleias12bhb": [("42", "pleias12bhb64"), ("62", "pleias12bhb64s62"), ("72", "pleias12bhb64s72")],
    "pleias3bhb": [("42", "pleias3bhb64"), ("62", "pleias3bhb64s62"), ("72", "pleias3bhb64s72")],
    "kl3m17bhb": [("42", "kl3m17bhb64"), ("62", "kl3m17bhb64s62"), ("72", "kl3m17bhb64s72")],
    "kl3m37bhb": [("42", "kl3m37bhb64"), ("62", "kl3m37bhb64s62")],
    "comma1t": [("42", "comma1thb64"), ("52", "comma1t64s52"), ("62", "comma1t64s62")],
}
LABELS = {"pleias12bhb": "Pleias-1.2B", "pleias3bhb": "Pleias-3B", "kl3m17bhb": "KL3M-1.7B",
          "kl3m37bhb": "KL3M-3.7B", "comma1t": "Comma-7B (1T)"}
# feat-138's four within-host moves, on record, which this arm extends rather than replaces.
ON_RECORD_MOVES = [("Comma-7B (1T)", "42-52", 0.0110), ("Pleias-350M", "42-52", 0.0070),
                   ("KL3M-170M", "42-52", 0.0080), ("KL3M-520M", "42-52", 0.0180)]


def read_draw(out, tag):
    """(delta, half_widths, n) for one draw, or None if its arm has not finished."""
    summ = rows(os.path.join(out, f"selection_scaling_{tag}.csv"))
    per = rows(os.path.join(out, f"selection_scaling_per_prompt_{tag}.csv"))
    ok, _msg = i3_coverage(summ, per)
    if not ok:
        return None
    g, lo, hi, hw, ratio, n = band([r for r in per if JUDGE_B in r["judge"]])
    return dict(delta=g, lo=lo, hi=hi, ratio=ratio, n=n, verdict=verdict_of(lo, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = a.out
    print("results/onset_prediction_within_host_spread.md (feat-141) -- is a host change larger")
    print("than a re-draw, or did three marginal anchors just wobble?")
    print(f"  imported: MAX_SEED_MOVE={MAX_SEED_MOVE}, boundary={HALF_WIDTHS_FOR_STABLE}")
    print(f"  P1 refutes feat-140 if any within-host pair at Pleias-3B reaches {REFUTES_AT}")

    all_moves = list(ON_RECORD_MOVES)
    recs, refuted = [], []
    for name, draws in DRAWS.items():
        label = LABELS[name]
        got = [(s, read_draw(out, t)) for s, t in draws]
        have = [(s, d) for s, d in got if d is not None]
        print(f"\n--- {label} ({name}): {len(have)} of {len(draws)} draws finished")
        for s, d in have:
            print(f"    seeds {s}x: {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
                  f"{d['ratio']:.2f} hw  {d['verdict']}")
        if len(have) < 2:
            print("    fewer than two draws; no within-host move can be formed")
            continue
        for (s1, d1), (s2, d2) in itertools.combinations(have, 2):
            mv = abs(d1["delta"] - d2["delta"])
            flag = ""
            if mv > MAX_SEED_MOVE:
                flag = f"  > {MAX_SEED_MOVE} (beyond every seed move on record)"
            if name == "pleias3bhb" and mv >= REFUTES_AT:
                flag += "  <<< REFUTES feat-140"
                refuted.append((label, f"{s1}-{s2}", mv))
            print(f"    within-host move {s1}-{s2}: {mv:.4f}{flag}")
            all_moves.append((label, f"{s1}-{s2}", mv))
            recs.append(dict(anchor=label, name=name, pair=f"{s1}-{s2}", move=round(mv, 4),
                             kind="within-host"))

    print("\n=== the within-host distribution, and where the cross-host moves sit in it ===")
    vals = sorted(m for _, _, m in all_moves)
    if vals:
        print(f"  {len(vals)} within-host moves: min {vals[0]:.4f}, median "
              f"{vals[len(vals) // 2]:.4f}, max {vals[-1]:.4f}")
        for nm, ch in sorted(CROSS_HOST.items(), key=lambda kv: -kv[1]):
            n_below = sum(1 for v in vals if v < ch)
            print(f"  cross-host {LABELS[nm]:<14} {ch:.4f}: larger than {n_below} of {len(vals)} "
                  f"within-host moves ({100 * n_below / len(vals):.0f}%)")
            recs.append(dict(anchor=LABELS[nm], name=nm, pair="cross-host", move=ch,
                             kind="cross-host"))

    # THE REGISTERED READING RULE, which is not the same test as P1's threshold and which the
    # pre-registration states in its own words: "The sentence 'a host change is not a re-draw'
    # survives only if the cross-host moves sit in the upper tail; if they sit in the body, it is
    # withdrawn in the same words it was written." Applied here so that a threshold surviving by a
    # hair cannot be reported as survival while the distribution says otherwise.
    UPPER_TAIL = 0.80
    if vals:
        print("\n=== the registered reading rule: tail or body? ===")
        in_body = []
        for nm, ch in CROSS_HOST.items():
            frac = sum(1 for v in vals if v < ch) / len(vals)
            where = "upper tail" if frac >= UPPER_TAIL else "BODY"
            print(f"  {LABELS[nm]:<14} {ch:.4f} at the {100 * frac:.0f}th percentile -> {where}")
            if frac < UPPER_TAIL:
                in_body.append(LABELS[nm])
        if in_body:
            print(f"  {len(in_body)} of {len(CROSS_HOST)} cross-host moves sit in the BODY "
                  f"({', '.join(in_body)}).")
            print("  By the rule committed before these data existed, the sentence")
            print("  \"a host change is not a re-draw\" is WITHDRAWN.")
        else:
            print("  All cross-host moves sit in the upper tail; the sentence survives.")

    print("\n=== P1 ===")
    if refuted:
        print("  REFUTED. feat-140's conclusion is WITHDRAWN: a within-host pair at Pleias-3B")
        print(f"  moved {refuted[0][2]:.4f}, at or beyond its cross-host move, so the host change")
        print("  is inside that anchor's own re-draw spread and the flips are wobble.")
    elif any(n == "pleias3bhb" for n, _ in
             [(r["name"], r) for r in recs if r["kind"] == "within-host"]):
        worst = max((r["move"] for r in recs
                     if r["kind"] == "within-host" and r["name"] == "pleias3bhb"), default=None)
        print(f"  SURVIVES so far: the largest within-host move at Pleias-3B is {worst:.4f}, "
              f"below the {REFUTES_AT} that would withdraw the conclusion.")
    else:
        print("  NOT READ: Pleias-3B has fewer than two finished draws.")

    c1 = [d for _, d in [(s, read_draw(out, t)) for s, t in DRAWS["comma1t"]] if d is not None]
    print("\n=== P3: does the one stable anchor stay stable? ===")
    if len(c1) >= 3:
        stable = all(d["ratio"] >= HALF_WIDTHS_FOR_STABLE and d["verdict"] == "CLIMBS" for d in c1)
        print(f"  Comma-7B (1T) over {len(c1)} draws: "
              + ", ".join(f"{d['ratio']:.2f} hw {d['verdict']}" for d in c1))
        print("  P3 " + ("HOLDS" if stable else "FAILS -- the only stable anchor is on the boundary"))
    else:
        print(f"  NOT READ: {len(c1)} of 3 draws finished.")

    p = os.path.join(out, "within_host_spread_scoring.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["anchor", "name", "pair", "move", "kind"])
        w.writeheader()
        for r in recs:
            w.writerow(r)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
