"""feat-180: the anchor-vetting screen as a function of prefix length.

Bands, gates and readings are results/onset_prediction_vetting_ladder.md; this script computes them.
Every rung is analysis/selection_extraction.py at the committed screen protocol with only
--seed-tokens changed, written as vetladder_L<L>_<tag>_per_passage.csv. The L = 100 rung of each
model is the arm on record (vet_<tag>, or selection_extraction_70b_hp2 for the 70B and the audited
anchor), read from its file, never retyped (caution (at)).

The screen's statistic is anchor_vetting.py's: a passage LEAKS when recall > 0 -- the max over 64
anchor draws for a model in the anchor slot, the single draw for the 70B, which only fits in the
risky slot.

No GPU.  Usage:  .venv/bin/python analysis/vetting_ladder.py --results results --out results
"""
import argparse
import csv
import glob
import os
import re

# model tag -> (statistic column, file holding its L = 100 rung, role)
MODELS = {
    "llama70b": ("risky_alone_recall", "selection_extraction_70b_hp2", "memorised in pre-training"),
    "olmo2_13b": ("anchor_max_recall", "vet_olmo2_13b", "open data"),
    "olmo2_7b": ("anchor_max_recall", "vet_olmo2_7b", "open data"),
    "tinycomma": ("anchor_max_recall", "selection_extraction_70b_hp2", "openly licensed"),
    "comma7b": ("anchor_max_recall", "vet_comma7b", "openly licensed"),
    "comma1t": ("anchor_max_recall", "vet_comma1t", "openly licensed"),
    "kl3m17b": ("anchor_max_recall", "vet_kl3m17b", "openly licensed"),
    "pleias12b": ("anchor_max_recall", "vet_pleias12b", "openly licensed"),
    "pleias3b": ("anchor_max_recall", "vet_pleias3b", "openly licensed"),
}
DROP = 3            # V1: a fall of 3 or more of 50 passages between adjacent rungs is NON-MONOTONE
SEES = 5            # V2: the screen "sees" the 70B once 5 of 50 passages leak


def load(path):
    with open(path, encoding="utf-8") as fh:
        return {r["prompt_id"]: r for r in csv.DictReader(fh)}


def g1_reproduction(results):
    """G1: the 70B's L = 100 re-run on this host IS the arm on record, passage for passage"""
    new = os.path.join(results, "vetladder_L100_llama70b_per_passage.csv")
    if not os.path.exists(new):
        return "NOT RUN"
    a, b = load(new), load(os.path.join(results, "selection_extraction_70b_hp2_per_passage.csv"))
    diff = [p for p in b if a.get(p, {}).get("risky_alone_recall") != b[p]["risky_alone_recall"]]
    return f"FAIL on {len(diff)} of {len(b)} passages" if diff else "PASS"


def rungs(results, use_record):
    """{tag: {L: (rows, source)}}: the new rungs, plus each model's L = 100 arm on record -- but only
    if G1 passed, because otherwise the host has drifted and the two are not one instrument"""
    out = {}
    for path in glob.glob(os.path.join(results, "vetladder_L*_per_passage.csv")):
        m = re.fullmatch(r"vetladder_L(\d+)_(.+)_per_passage\.csv", os.path.basename(path))
        out.setdefault(m.group(2), {})[int(m.group(1))] = (load(path), "this arm")
    for tag, (_, ref, _) in MODELS.items():
        if use_record and tag in out and 100 not in out[tag]:
            out[tag][100] = (load(os.path.join(results, f"{ref}_per_passage.csv")), "on record")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    g1 = g1_reproduction(a.results)
    ladder = rungs(a.results, use_record=(g1 == "PASS"))
    corpus = set(load(os.path.join(a.results, "vet_comma7b_per_passage.csv")))
    rows, g0 = [], []
    for tag, by in sorted(ladder.items()):
        col, _, role = MODELS[tag]
        for L in sorted(by):
            data, source = by[L]
            if set(data) != corpus:         # G0: not the 50 passages -> not read
                g0.append(f"{tag} L={L}")
                continue
            v = [float(r[col]) for r in data.values()]
            rows.append(dict(model=tag, role=role, prefix_tokens=L, statistic=col,
                             n_passages=len(v), leaking=sum(x > 0 for x in v),
                             frac_leaking=round(sum(x > 0 for x in v) / len(v), 4),
                             max_recall=round(max(v), 4), mean_recall=round(sum(v) / len(v), 4),
                             source=source))
    print(f"  G0 {'PASS' if not g0 else 'FAIL, not read: ' + ', '.join(g0)}   G1 {g1}")
    if not rows:
        print("  no readable rung\n\n  V3 NOT READ")
        return
    path = os.path.join(a.out, "vetting_ladder.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['model']:10s} L={r['prefix_tokens']:>3}  leaks on {r['leaking']:2d}/50  "
              f"max {r['max_recall']:.4f}  [{r['source']}]")

    curve = lambda t: [(r["prefix_tokens"], r["leaking"]) for r in rows if r["model"] == t]  # noqa
    dirty = sorted({r["model"] for r in rows if r["role"] != "openly licensed"})
    falls = {t: [(l1, l2) for (l1, k1), (l2, k2) in zip(curve(t), curve(t)[1:]) if k1 - k2 >= DROP]
             for t in dirty}
    v1 = "NON-MONOTONE" if any(falls.values()) else "MONOTONE"
    seen = [L for L, k in curve("llama70b") if k >= SEES]
    v2 = f"L* = {min(seen)}" if seen else "NOT SEEN AT ANY RUNG"
    long = [r for r in rows if r["role"] == "openly licensed" and r["prefix_tokens"] > 100]
    v3 = ("NOT READ" if not long else
          "PASS BREAKS" if any(r["leaking"] for r in long) else "PASS HOLDS")
    v4 = ("screen at the longest prefix the deployment accepts" if v1 == "MONOTONE"
          else "screen at every rung: no single prefix length dominates")
    print(f"\n  V1 {v1} {falls if v1 != 'MONOTONE' else ''}\n  V2 {v2}\n  V3 {v3}\n  V4 {v4}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
