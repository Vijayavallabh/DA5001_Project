"""Plan v5 / feat-060: is the tokenizer split a property of the decoder or of the metric?

The six measured pairs fall into two groups by onset ratio, and the grouping tracks how many
characters the tokenizer packs into a token. Before that can be reported as a fact about metered
decoding it has to survive the metric, because the headline metric is normalised in a way that
could produce it: nv_recall is `matched reference words / reference words`, and the KL3M pairs have
the same reference words as everyone else but twice the decode steps.

Three metrics, each normalised differently, swept over their thresholds:

  nv_recall   fraction of REFERENCE WORDS inside a >= 20-word verbatim span (the paper's metric)
  lcs_word    longest common subsequence in WORDS, an absolute count with no denominator at all
  any_span    fraction of PASSAGES with any qualifying span -- denominator is passages, not words

A split that appears under all three at every threshold both groups cross is a property of the
pairs. One that appears under a single normalisation is a property of the metric.

Writes <out>/split_robustness.csv. No GPU.

Usage: .venv/bin/python analysis/split_robustness.py --out results
"""
import argparse, collections, csv, os, statistics as st, sys

# (label, per-passage composition.csv, s(x) nats/token, characters per token)
PAIRS = [
    ("TinyComma-1.8B", "output/phase4/fine_tc/composition.csv", 3.2386, 4.20),
    ("Comma-7B", "output/phase4/fine_comma/composition.csv", 2.3935, 3.63),
    ("Pleias-350M", "output/phase5/fine_pleias350m/composition.csv", 3.5544, 4.05),
    ("Pleias-1.2B", "output/phase5/fine_pleias12b/composition.csv", 3.2094, 4.05),
    ("KL3M-520M", "output/phase5/fine_kl3m520m/composition.csv", 2.4153, 1.96),
    ("KL3M-1.7B", "output/phase5/fine_kl3m17b_merged/composition.csv", 2.2112, 1.96),
]
METRICS = {
    "nv_recall": [0.005, 0.01, 0.02, 0.03, 0.05],
    "lcs_word": [4.0, 5.0, 6.0, 8.0, 10.0],
    "any_span": [0.02, 0.03, 0.05, 0.08],
}
GROUP_SPLIT_CHARS_PER_TOKEN = 3.0


def crossing(curve, thresh):
    """First budget at which the curve reaches `thresh`, interpolated inside its bracket.

    None when the curve never reaches the threshold, and also when the SMALLEST budget probed is
    already at or above it: there is no bracket then, and interpolating from one anyway
    extrapolates backwards and returns a budget below the grid. analysis/onset.py has always
    treated that case as "at/below grid" rather than a measurement; this now matches it.
    """
    prev = None
    for k in sorted(curve):
        if curve[k] >= thresh:
            if prev is None:
                return None
            lo, hi = curve[prev], curve[k]
            return prev + (k - prev) * (thresh - lo) / (hi - lo) if hi > lo else k
        prev = k
    return None


def curves(path):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["mode"] != "single" or float(r["k"]) <= 0:
            continue
        k, nv = float(r["k"]), float(r["nv_recall"])
        by["nv_recall"][k].append(nv)
        by["lcs_word"][k].append(float(r["lcs_word"]))
        by["any_span"][k].append(1.0 if nv > 0 else 0.0)
    return {m: {k: st.mean(v) for k, v in d.items()} for m, d in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    data = {}
    for name, f, s_x, cpt in PAIRS:
        if not os.path.exists(f):
            print(f"[split] missing {f}", file=sys.stderr)
            continue
        data[name] = (s_x, cpt, curves(f))
    if len(data) < 4:
        raise SystemExit("[split] need the full pair set")

    rows = []
    for metric, thresholds in METRICS.items():
        for t in thresholds:
            ratios, incomplete = {}, False
            for name, (s_x, cpt, cs) in data.items():
                o = crossing(cs[metric], t)
                if o is None:
                    incomplete = True
                    break
                ratios[name] = (o / s_x, cpt)
            if incomplete:
                rows.append({"metric": metric, "threshold": t, "n_pairs": len(data),
                             "wide_lo": "", "wide_hi": "", "narrow_lo": "", "narrow_hi": "",
                             "gap": "", "verdict": "a pair never crosses"})
                continue
            wide = [r for r, c in ratios.values() if c > GROUP_SPLIT_CHARS_PER_TOKEN]
            narrow = [r for r, c in ratios.values() if c <= GROUP_SPLIT_CHARS_PER_TOKEN]
            gap = min(narrow) - max(wide)
            rows.append({"metric": metric, "threshold": t, "n_pairs": len(data),
                         "wide_lo": round(min(wide), 4), "wide_hi": round(max(wide), 4),
                         "narrow_lo": round(min(narrow), 4), "narrow_hi": round(max(narrow), 4),
                         "gap": round(gap, 4), "verdict": "split" if gap > 0 else "overlap"})

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "split_robustness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("onset ratio by tokenizer group (> 3 characters per token vs <= 3), per metric\n")
    print(f"{'metric':12s}{'thresh':>8s}{'>3 ch/tok':>16s}{'<=3 ch/tok':>16s}{'gap':>9s}  verdict")
    for r in rows:
        if r["verdict"].startswith("a pair"):
            print(f"{r['metric']:12s}{r['threshold']:8g}{'':>32s}{'':>9s}  {r['verdict']}")
            continue
        wide = f"{r['wide_lo']:.3f}-{r['wide_hi']:.3f}"
        narrow = f"{r['narrow_lo']:.3f}-{r['narrow_hi']:.3f}"
        print(f"{r['metric']:12s}{r['threshold']:8g}"
              f"{wide:>16s}{narrow:>16s}{r['gap']:+9.3f}  {r['verdict']}")
    usable = [r for r in rows if r["verdict"] in ("split", "overlap")]
    n_split = sum(r["verdict"] == "split" for r in usable)
    print(f"\n{n_split}/{len(usable)} usable (metric, threshold) cells show the split")
    for m in METRICS:
        sub = [r for r in usable if r["metric"] == m]
        if sub:
            print(f"  {m:10s} {sum(r['verdict']=='split' for r in sub)}/{len(sub)}")
    print(f"\nwrote {a.out}/split_robustness.csv")


if __name__ == "__main__":
    main()
