"""Does the constructive claim hold at more than one anchor?

Every selection-anchoring number in the paper comes from TinyComma-1.8B. The v7 reframe made that
half load-bearing, so one anchor is now the narrowest evidence we have. This aggregates the
per-anchor sweeps -- each produced by the SAME pointwise-reward and two-judge pass as feat-088, so
nothing but the anchor changes -- and scores them against the bands committed in
results/onset_prediction_selection_breadth.md before any of them was generated.

Reads:  <out>/selection_scaling{,_<tag>}.csv, one per anchor
Writes: <out>/selection_breadth.csv

Usage: .venv/bin/python analysis/selection_breadth.py --out results
"""
import argparse
import csv
import os

ANCHORS = [("TinyComma-1.8B (audited)", "", "jacquelinehe/tinycomma-1.8b-llama3-tokenizer"),
           ("Pleias-1.2B", "_pleias12b", "PleIAs/Pleias-1.2b-Preview"),
           ("KL3M-1.7B", "_kl3m17b", "alea-institute/kl3m-003-1.7b"),
           ("Comma-7B", "_comma7b", "common-pile/comma-v0.1-2t")]
GATE_TOKENS = 20.0          # the entry gate: an anchor that cannot write is not evidence


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(o):
            r[i] = pos + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    rows = []
    for label, tag, model in ANCHORS:
        path = os.path.join(a.out, f"selection_scaling{tag}.csv")
        if not os.path.exists(path):
            print(f"  [breadth] missing {path}, skipping {label}")
            continue
        r = list(csv.DictReader(open(path)))
        # the scored (not selecting) judge, at n = 1 and n = a.n
        judges = sorted({x["judge"] for x in r if x.get("judge")})
        for j in judges:
            arm = {int(float(x["n"])): x for x in r if x.get("judge") == j}
            if 1 not in arm or a.n not in arm:
                continue
            one, many = arm[1], arm[a.n]
            tokens = float(one.get("mean_tokens") or 0)
            gate = tokens >= GATE_TOKENS
            gain = float(many["u"]) - float(one["u"])
            lo = float(many.get("gain_lo95") or "nan")
            hi = float(many.get("gain_hi95") or "nan")
            rows.append(dict(anchor=label, model=model, judge=j, n=a.n,
                             u_n1=round(float(one["u"]), 4), u_n=round(float(many["u"]), 4),
                             gain=round(gain, 4), gain_lo95=lo, gain_hi95=hi,
                             mean_tokens_n1=round(tokens, 1),
                             entry_gate="PASS" if gate else "FAIL"))
    if not rows:
        print("  [breadth] nothing to aggregate yet")
        return 1
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "selection_breadth.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    for r in rows:
        ci = (f"[{r['gain_lo95']:+.4f}, {r['gain_hi95']:+.4f}]"
              if r["gain_lo95"] == r["gain_lo95"] else "[no CI]")
        print(f"  {r['anchor']:26s} {r['judge'][:28]:30s} u {r['u_n1']:.3f} -> {r['u_n']:.3f}  "
              f"gain {r['gain']:+.4f} {ci}  gate {r['entry_gate']}")

    # B1: how many of the NEW anchors have a gain whose CI excludes 0, on the scoring judge
    new = [r for r in rows if "audited" not in r["anchor"] and r["entry_gate"] == "PASS"]
    by_anchor = {}
    for r in new:
        by_anchor.setdefault(r["anchor"], []).append(r)
    excl = sum(1 for v in by_anchor.values()
               if any(x["gain_lo95"] == x["gain_lo95"] and x["gain_lo95"] > 0 for x in v))
    b1 = ("GENERALISES" if excl >= 2 else "PARTIAL" if excl == 1 else "SINGLE-SETUP")
    print(f"\n  B1 anchors with a gain CI excluding 0: {excl}/{len(by_anchor)}  ->  {b1}")
    if b1 == "SINGLE-SETUP":
        print("     the constructive claim must be narrowed to the audited pair; see the "
              "pre-registration for the exact edits that forces.")
    # B2: descriptive only -- four points cannot establish a ceiling
    per = {}
    for r in rows:
        per.setdefault(r["anchor"], []).append(r)
    xs = [sum(x["u_n1"] for x in v) / len(v) for v in per.values()]
    ys = [sum(x["gain"] for x in v) / len(v) for v in per.values()]
    if len(xs) >= 3:
        print(f"  B2 Spearman(anchor-alone u, gain) = {spearman(xs, ys):+.3f} over {len(xs)} "
              f"anchors -- descriptive, not evidence for the ceiling argument")
    print("  B3 leakage is scored by analysis/selection_extraction.py per anchor; any non-zero "
          "recall goes in the main text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
