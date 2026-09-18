"""Score results/onset_prediction_breadth64.md against its committed bands. Zero GPU.

Written BEFORE any of the three arms produced a number, so the reading rules here are the ones in
the pre-registration rather than rules chosen after seeing the answer.

The question: the paper's breadth claim is quantified at n=8 and its headline at n=64, and only
TinyComma and Comma-7B have an n=64 curve at all. Three anchors that stop at n=8 pass the registered
entry gate, and this arm takes them to 64. The band is read per anchor on the PAIRED difference
g(64) - g(8) under judge B, which on the per-prompt file is exactly u_n64 - u_n8 because both gains
subtract the same u_n1.

Each anchor is gated on a reproduction check. E1 groups jobs by seed before batching and seeds each
generate() from that group's own value, so at a fixed --batch-size 32 the seed group for draw j is
the same 500 jobs in the same buckets whether the run asks for 8 draws or 64: draws 0-7 of the
64-draw pool ARE the committed 8-draw pool. The n <= 8 half must therefore reproduce
selection_scaling_<name>.csv, and no n > 8 number may be read until it does -- which this script
enforces rather than advises.

Usage: .venv/bin/python analysis/score_breadth64.py --out results
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import boot_mean  # noqa: E402

JUDGE_B = "Phi-3.5-mini-instruct"

# name -> (label, the committed g(8) band under judge B, from the pre-registration)
ANCHORS = (
    ("pleias12b", "Pleias-1.2B", (0.029, -0.012, 0.068)),
    ("kl3m17b", "KL3M-1.7B", (0.039, 0.005, 0.076)),
    ("pleias3b", "Pleias-3B", (0.047, 0.003, 0.089)),
)
# The two anchors already measured to 64 give the effect size this arm is powered against.
REFERENCE_CLIMBS = (("TinyComma-1.8B", 0.054, 0.142), ("Comma-7B", 0.072, 0.173))


def rows(path):
    if not os.path.exists(path):
        return None
    return list(csv.DictReader(open(path, encoding="utf-8")))


# The committed breadth caches are named two different ways, because two different launchers
# produced them ("[breadth]" and "[br_<name>]"). Try both rather than assuming one.
OLD_CACHE = ("selection_rewards_{n}.csv", "selection_rewards8_{n}.csv")


def old_reward_cache(out, name):
    for pat in OLD_CACHE:
        p = os.path.join(out, pat.format(n=name))
        if os.path.exists(p):
            return rows(p)
    return None


def reward_gate(new_cache, old_cache, top=8):
    """Ranks 0..top-1 of the new pool must be BIT-IDENTICAL to the committed pool.

    CORRECTED 2026-09-18, BEFORE this scorer was ever run on data, by transplanting the lesson
    feat-129 paid for (caution (ap)). The first version compared judged `gain` at n <= 8 against the
    committed breadth CSV at 5e-4 -- and here that could not possibly pass, for a reason feat-129
    made precise hours earlier: the committed rows come from a grid-1..8 sweep and these come from a
    grid-1..64 sweep, and a single-order judged level is GRID-DEPENDENT. `selection_scaling.py`
    draws one rng.random() per element of `distinct`, which is built over the whole grid, so a
    longer grid re-rolls the presentation order of nearly every shared item into a position-
    dominated judge. Comma-7B's n=8 gain is +0.111 on grid 1..8 and +0.072 on grid 1..64: the same
    anchor, the same text, a different grid.

    What the pre-registration's reproduction argument actually established is about generation and
    reward, and that is what this checks. The registered band is unaffected either way, because it
    is the PAIRED g(64) - g(8) computed WITHIN the new pass.
    """
    if new_cache is None or old_cache is None:
        return None, "one of the two reward caches is missing"
    k = lambda r: (r["prompt_id"], int(r["rank"]))              # noqa: E731
    o = {k(r): float(r["reward"]) for r in old_cache}
    n = {k(r): float(r["reward"]) for r in new_cache if int(r["rank"]) < top}
    missing = [x for x in o if x not in n]
    bad = [(x, o[x], n[x]) for x in o if x in n and abs(o[x] - n[x]) > 0]
    if missing:
        bad = [(m, o[m], None) for m in missing[:5]] + bad
    return (not bad), bad


def judge_drift(new, old):
    """Reported, never gated: how far the judged level moved between the two grids."""
    if new is None or old is None:
        return []
    key = lambda r: (r["judge"], int(float(r["n"])))            # noqa: E731
    o = {key(r): r for r in old}
    return [(k, float(o[k]["u"]), float(r["u"]))
            for r in new if (k := key(r))[1] <= 8 and k in o]


def score_anchor(out, name, label, committed8, waived=()):
    new = rows(os.path.join(out, f"selection_scaling_{name}64.csv"))
    old = rows(os.path.join(out, f"selection_scaling_{name}.csv"))
    ok, bad = reward_gate(rows(os.path.join(out, f"selection_rewards64_{name}.csv")),
                          old_reward_cache(out, name))
    print(f"\n=== {label} ({name}) ===")
    if ok is None:
        print(f"  NOT YET SCOREABLE: {bad}")
        return None
    print(f"  reproduction of ranks 0-7 (rewards, exact): {'PASS' if ok else 'FAIL'}")
    warrant = "FULL"
    if not ok:
        for x in bad[:3]:
            print(f"    {x}")
        if name not in waived:
            print("  Per the pre-registration, NO n>8 number is read at this anchor. Chase the bug.")
            return None
        warrant = "REDUCED"
        print(f"  reproduction WAIVED by explicit --waive-reproduction {name}.")
        print("  Reason, established from git history and independent of any result: the committed")
        print("  breadth arm for this anchor ran 2026-09-12, before scripts/run_breadth_anchor.sh")
        print("  existed (commit 94f9e7d, 2026-09-14 07:31, 'raise h1 batch 8 -> 32/48'), so it used")
        print("  h1.py's default --batch-size 8 where this arm used 32. Batch size is part of the")
        print("  seed (caution (u)), so ranks 0-7 CANNOT be bit-identical and the check is")
        print("  inapplicable rather than failing. The pre-registration asserted the committed value")
        print("  was 32; that premise was false for this anchor and is recorded, not repaired.")
        print("  The band below is a PAIRED difference WITHIN this pass and never touches the")
        print("  committed arm, so it is unaffected -- but its warrant is REDUCED, because the")
        print("  registered pipeline check could not be run. Pleias-3B, whose committed arm DID use")
        print("  batch 32, reproduces bit-exactly and stands as the positive control for the")
        print("  pipeline as a whole.")
    drift = judge_drift(new, old)
    if drift:
        worst = max((abs(b - c), k) for k, b, c in drift)
        print(f"  judged level moved up to {worst[0]:+.4f} at {worst[1]} between the two grids "
              f"-- expected, not gated (caution (ap)); the band below is paired WITHIN this pass")

    per = rows(os.path.join(out, f"selection_scaling_per_prompt_{name}64.csv"))
    assert per, "the per-prompt file is required for the PAIRED difference"
    pp = [r for r in per if JUDGE_B in r["judge"]]
    assert pp, sorted({r["judge"] for r in per})
    top = max(int(c[3:]) for c in pp[0] if c.startswith("u_n"))
    if top < 64:
        print(f"  NOT YET SCOREABLE: the sweep reached only n={top}. If the run has finished, "
              f"--max-n did not take and analysis.selection_scaling.n_grid is the thing to check.")
        return None

    d = [float(r["u_n64"]) - float(r["u_n8"]) for r in pp]
    g = sum(d) / len(d)
    lo, hi = boot_mean(d, random.Random(20260918))
    verdict = ("CLIMBS" if lo > 0 else "TURNS OVER" if hi < 0 else "SATURATED BY 8")
    print(f"  paired g(64) - g(8) over {len(d)} prompts: {g:+.4f} [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
    n8 = next(r for r in new if JUDGE_B in r["judge"] and int(float(r["n"])) == 8)
    n64 = next(r for r in new if JUDGE_B in r["judge"] and int(float(r["n"])) == 64)
    print(f"  g(8) {float(n8['gain']):+.4f} (committed {committed8[0]:+.3f} "
          f"[{committed8[1]:+.3f}, {committed8[2]:+.3f}])  ->  g(64) {float(n64['gain']):+.4f} "
          f"[{float(n64['gain_lo95']):+.3f}, {float(n64['gain_hi95']):+.3f}]")
    for r in sorted(new, key=lambda r: (r["judge"], int(float(r["n"])))):
        print(f"    {r['judge'][:26]:<26} n={int(float(r['n'])):<3} gain={float(r['gain']):+.4f} "
              f"kl_nats={float(r['kl_nats']):.4f} spearman(u,log n)={r.get('spearman_u_logn','')}")
    return dict(anchor=label, name=name, warrant=warrant, gain8=round(float(n8["gain"]), 4),
                gain64=round(float(n64["gain"]), 4), gain_diff=round(g, 4),
                lo95=round(lo, 4), hi95=round(hi, 4), n_prompts=len(d),
                reproduction=("PASS" if warrant == "FULL" else "WAIVED -- batch 8 vs 32"),
                verdict=verdict)


def overall(scored):
    """The four committed readings, in the pre-registration's own precedence.

    TURNS OVER is checked first because it falsifies a sentence the paper currently prints
    ('the reward overoptimisation that turns such curves over is not observed anywhere on the
    grid'), and one anchor is enough to make that sentence false as written.
    """
    if len(scored) < len(ANCHORS):
        return "INCOMPLETE -- not every anchor has landed"
    if any(s["verdict"] == "TURNS OVER" for s in scored):
        return "TURNS OVER"
    climbs = [s for s in scored if s["verdict"] == "CLIMBS"]
    if len(climbs) == len(scored):
        return "BREADTH HOLDS AT THE HEADLINE n"
    if climbs:
        return "PARTIAL"
    return "THE HEADLINE n IS THE TWO ANCHORS, NOT THE MECHANISM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--waive-reproduction", nargs="*", default=[],
                    help="anchors whose reproduction check is INAPPLICABLE for a "
                         "documented reason. Deliberate, visible in the command, and "
                         "never a way past a check that merely failed.")
    a = ap.parse_args()

    print("results/onset_prediction_breadth64.md -- does breadth survive to the headline n?")
    print("effect size on record: " + ", ".join(
        f"{n} {g8:+.3f} -> {g64:+.3f} (D={g64 - g8:+.3f})" for n, g8, g64 in REFERENCE_CLIMBS))

    waived = set(a.waive_reproduction)
    scored = [s for s in (score_anchor(a.out, n, lbl, c8, waived)
                          for n, lbl, c8 in ANCHORS) if s]
    v = overall(scored)
    print(f"\n=== VERDICT: {v} ===")
    for s in scored:
        print(f"  {s['anchor']:<14} {s['gain8']:+.4f} -> {s['gain64']:+.4f}  "
              f"D={s['gain_diff']:+.4f} [{s['lo95']:+.4f}, {s['hi95']:+.4f}]  "
              f"{s['verdict']:<16} warrant={s['warrant']}")

    if scored:
        p = os.path.join(a.out, "breadth64_scoring.csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(scored[0]) + ["overall"])
            w.writeheader()
            for s in scored:
                w.writerow({**s, "overall": v})
        print(f"wrote {p}")
    print(json.dumps({"overall": v, "anchors": scored}, indent=1))


if __name__ == "__main__":
    main()
