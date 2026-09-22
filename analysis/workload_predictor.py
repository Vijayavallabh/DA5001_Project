"""feat-173 B3/B4: does the anchor's own competence predict where the reversal holds?

EXPLORATORY, and the registration says so. The hypothesis was committed before these numbers were
computed, which is the only thing that separates it from a story fitted to the data; it is still a
hypothesis test performed on arms that already existed, and it is never quoted as pre-registered.

The predictor is the anchor's own judged win rate against the unconstrained opponent at n=1 --- a
pure statement about whether the anchor can do the task, measurable before any selection or
metering is run. `u_anchor_k0` in each pass's per-prompt output is exactly that quantity.

Both sides come from the SAME pass for each workload, so no judged level crosses a pass boundary
(caution (ap)): the win rate and the paired difference are computed on one set of prompts under one
judge in one sweep.
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import paired_boot  # noqa: E402

# (label, per-prompt file, metered column, optional class filter)
WORKLOADS = [
    ("ours, all 850", "wscope_c", "u_metered_k0.9", None),
    ("ours: neutral", "wscope_c", "u_metered_k0.9", "neutral"),
    ("ours: creative", "wscope_c", "u_metered_k0.9", "creative"),
    ("ours: factual", "wscope_c", "u_metered_k0.9", "factual"),
    ("AlpacaEval", "mixpowk_judgeB", "u_metered_k1", None),
]


def load(tag, results):
    path = os.path.join(results, f"order_averaged_h2h_per_prompt__{tag}.csv")
    return list(csv.DictReader(open(path, encoding="utf-8")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--classes", required=True, help="prompt_id,cls map for the per-class split")
    ap.add_argument("--out", default="results/workload_predictor.csv")
    ap.add_argument("--seed", type=int, default=7717)
    a = ap.parse_args()

    cls = {r["prompt_id"]: r["cls"] for r in csv.DictReader(open(a.classes, encoding="utf-8"))}
    rng = random.Random(a.seed)
    rows = []
    for label, tag, mcol, want in WORKLOADS:
        recs = load(tag, a.results)
        if want:
            recs = [r for r in recs if cls.get(r["prompt_id"]) == want]
            assert recs, f"{label}: no prompts of class {want}"
        # The predictor: the anchor alone against the unconstrained opponent, n=1.
        anchor = sum(float(r["u_anchor_k0"]) for r in recs) / len(recs)
        diffs = [float(r["u_sel_n64"]) - float(r[mcol]) for r in recs]
        d3 = sum(diffs) / len(diffs)
        lo, hi = paired_boot(diffs, rng)
        sign = "selection" if lo > 0 else "meter" if hi < 0 else "unresolved"
        rows.append({"workload": label, "n_prompts": len(recs),
                     "anchor_win_rate_n1": round(anchor, 4),
                     "d3": round(d3, 4), "lo95": round(lo, 4), "hi95": round(hi, 4),
                     "winner": sign})

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'workload':<18}{'n':>6}{'anchor u(n=1)':>15}{'paired D3':>12}{'95% CI':>22}  winner")
    for r in rows:
        band = f"[{r['lo95']:+.4f}, {r['hi95']:+.4f}]"
        print(f"{r['workload']:<18}{r['n_prompts']:>6}{r['anchor_win_rate_n1']:>15.4f}"
              f"{r['d3']:>+12.4f}{band:>22}  {r['winner']}")

    # Does the predictor order the outcomes? Report it; do not assert it. A bare "the ranges
    # separate" boolean is far too weak a test with one workload on one side, so the dose-response
    # WITHIN our own corpus is reported beside it: three classes at one protocol in one pass,
    # spanning a wider range of anchor competence than the gap to AlpacaEval. If competence drives
    # the sign, D3 should track it there too.
    resolved = [r for r in rows if r["winner"] != "unresolved" and r["workload"] != "ours, all 850"]
    sel = [r["anchor_win_rate_n1"] for r in resolved if r["winner"] == "selection"]
    met = [r["anchor_win_rate_n1"] for r in resolved if r["winner"] == "meter"]
    print()
    if sel and met:
        print(f"anchor win rate where SELECTION wins: {min(sel):.4f}-{max(sel):.4f}  (n={len(sel)})")
        print(f"anchor win rate where METER wins:     {min(met):.4f}-{max(met):.4f}  (n={len(met)})")
        print(f"ranges separate: {min(sel) > max(met)}   gap {min(sel) - max(met):+.4f}")
    else:
        print("only one outcome class among the resolved workloads; the predictor is untestable "
              "here and that is the finding")

    within = [r for r in rows if r["workload"].startswith("ours: ")]
    if len(within) >= 3:
        xs = [r["anchor_win_rate_n1"] for r in within]
        ys = [r["d3"] for r in within]
        span_in = max(xs) - min(xs)
        spread = max(ys) - min(ys)
        gap = (min(sel) - max(met)) if (sel and met) else float("nan")
        print(f"\nWITHIN our corpus, three classes at one protocol in one pass:")
        for r in within:
            print(f"   {r['workload']:<18} anchor {r['anchor_win_rate_n1']:.4f}  "
                  f"D3 {r['d3']:+.4f}  {r['winner']}")
        print(f"   competence span {span_in:.4f} (vs the cross-workload gap {gap:+.4f}), "
              f"D3 spread {spread:.4f}, signs {'ALL THE SAME' if len({r['winner'] for r in within}) == 1 else 'DIFFER'}")
        order = [y for _x, y in sorted(zip(xs, ys))]
        mono = all(b >= a for a, b in zip(order, order[1:]))
        print(f"   D3 monotone in anchor competence within the corpus: {mono}")
        if span_in > gap and len({r["winner"] for r in within}) == 1:
            print("   => the predictor has NO dose-response where we can see it best: a WIDER "
                  "competence range inside one pass moves the sign not at all.")


if __name__ == "__main__":
    main()
