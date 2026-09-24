"""Read results/forest_deecho_note.md: each judged forest row on record against its re-judge. No GPU.

A row survives if its sign class (POSITIVE lo > 0, NEGATIVE hi < 0, else COVERS ZERO) is unchanged.
Writes results/forest_rejudge.csv.
Usage: .venv/bin/python analysis/forest_rejudge_score.py --out results
"""
import argparse
import csv
import os

SWEEPS = (("audited", "", (8, 64)), ("Pleias-1.2B", "_pleias12b", (8,)), ("KL3M-1.7B", "_kl3m17b", (8,)),
          ("Comma-7B", "_comma7b", (8,)), ("Comma-7B (1T)", "_comma1t", (8,)),
          ("Pleias-3B", "_pleias3b", (8,)), ("AlpacaEval, TinyComma", "_alpaca", (8,)),
          ("AlpacaEval, Comma-7B", "_alpaca_comma7b", (8,)), ("MT-Bench, TinyComma", "_mtbench", (8,)))


def cls(lo, hi):
    return "POSITIVE" if lo > 0 else "NEGATIVE" if hi < 0 else "COVERS ZERO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = []
    for name, tag, ns in SWEEPS:
        new = os.path.join(a.out, f"selection_scaling{tag}_deecho.csv")
        if not os.path.exists(new):
            print(f"[forest] {name}: no re-judge yet")
            continue
        old = {(r["judge"], int(float(r["n"]))): r for r in csv.DictReader(open(os.path.join(a.out, f"selection_scaling{tag}.csv")))}
        for r in csv.DictReader(open(new)):
            n = int(float(r["n"]))
            if n not in ns:
                continue
            o = old[(r["judge"], n)]
            g0, l0, h0 = (float(o[k]) for k in ("gain", "gain_lo95", "gain_hi95"))
            g1, l1, h1 = (float(r[k]) for k in ("gain", "gain_lo95", "gain_hi95"))
            out.append(dict(sweep=name, judge=r["judge"].split("/")[-1], n=n,
                            gain_record=g0, lo_record=l0, hi_record=h0, reading_record=cls(l0, h0),
                            gain_deecho=g1, lo_deecho=l1, hi_deecho=h1, reading_deecho=cls(l1, h1),
                            survives="YES" if cls(l0, h0) == cls(l1, h1) else "NO"))
    path = os.path.join(a.out, "forest_rejudge.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['sweep']:24s} {r['judge'][:12]:12s} n={r['n']:<3d} {r['gain_record']:+.4f} "
              f"[{r['lo_record']:+.4f}, {r['hi_record']:+.4f}] -> {r['gain_deecho']:+.4f} "
              f"[{r['lo_deecho']:+.4f}, {r['hi_deecho']:+.4f}]  {r['reading_deecho']:12s} {r['survives']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
