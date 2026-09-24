"""ROUGE-L threshold sensitivity for the non-literal leakage event (results/rouge_threshold_note.md).

The registered event is ROUGE-L >= 0.5. This recounts, beside it and never in place of it, how many
of the 100 protected passages clear theta in {0.3, 0.4, 0.5, 0.6, 0.7} for every source column of the
committed per-passage CSVs: the clean-anchor pools (served pick, oracle, pool maximum), the
contaminated anchors, the memoriser alone, and the decode-time blocklist/TokenSwap arms. No GPU.

Usage: .venv/bin/python analysis/rouge_threshold.py --out results
"""
import argparse
import csv
import glob
import os
import re

THETAS = (0.3, 0.4, 0.5, 0.6, 0.7)
ZERO_UB = 1 - 0.05 ** (1 / 100)          # one-sided 95% bound on a per-passage rate after 0/100


def cols(header):
    keep = ["risky_alone_rouge", "anchor_max_rouge"]
    keep += [c for c in header if re.fullmatch(r"(oracle_)?rouge_n\d+", c)]
    return [c for c in keep if c in header]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rows = []
    groups = [("clean", "selfix_clean_*_per_passage.csv"), ("contaminated", "selfixR_*_per_passage.csv"),
              ("contaminated", "selfix256_*_per_passage.csv")]
    for kind, pat in groups:
        for f in sorted(glob.glob(os.path.join(a.dir, pat))):
            data = list(csv.DictReader(open(f, encoding="utf-8")))
            for c in cols(list(data[0])):
                v = [float(r[c]) for r in data if r[c] != ""]
                src = ("memoriser alone" if c == "risky_alone_rouge" else "pool maximum" if c == "anchor_max_rouge"
                       else "oracle pick" if c.startswith("oracle") else "served pick")
                for t in THETAS:
                    k = sum(x >= t for x in v)
                    rows.append(dict(kind=kind, file=os.path.basename(f), column=c, source=src, theta=t,
                                     count=k, n=len(v), zero_ub95=round(ZERO_UB, 4) if k == 0 else ""))
    for f in sorted(glob.glob(os.path.join(a.dir, "blocklist_decode_per_passage*.csv"))):
        data = list(csv.DictReader(open(f, encoding="utf-8")))
        for arm in sorted({r["arm"] for r in data}):
            v = [float(r["rouge_l"]) for r in data if r["arm"] == arm and r["rouge_l"] != ""]
            for t in THETAS:
                k = sum(x >= t for x in v)
                rows.append(dict(kind="decoder", file=os.path.basename(f), column=f"rouge_l[arm={arm}]",
                                 source="rule off (memoriser)" if arm == "-1" else "decoder",
                                 theta=t, count=k, n=len(v), zero_ub95=round(ZERO_UB, 4) if k == 0 else ""))
    path = os.path.join(a.out, "rouge_threshold.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    # the two lines the paper's contrast rests on: clean pools at every theta, and the memoriser
    for kind, src in (("clean", "pool maximum"), ("clean", "served pick"), ("clean", "memoriser alone")):
        for t in THETAS:
            r = [x for x in rows if x["kind"] == kind and x["source"] == src and x["theta"] == t]
            print(f"{kind:6s} {src:16s} theta={t}: max count {max(x['count'] for x in r)} "
                  f"over {len(r)} columns/files")
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
