"""Recompute the `reading` column of every D3 row from its own numbers.

The label is a pure function of (point estimate, lo95, hi95) -- it is derived, never measured -- so
recomputing it changes no measurement and needs no GPU. It is done because
`analysis/order_averaged_h2h.py` emitted an asymmetric verdict until 2026-09-22: any negative point
estimate was called REVERSAL REFUTED whatever its interval did, while the mirror image was called
UNRESOLVED. Two rows on disk carry the old label.

Numbers are untouched. Run with --check to report disagreements without writing.
"""
import argparse
import csv
import glob
import os


def verdict(g, lo, hi):
    if lo <= 0 <= hi:
        return "REVERSAL UNRESOLVED"
    return "REVERSAL CONFIRMED" if g > 0 else "REVERSAL REFUTED"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/order_averaged_h2h__*.csv")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    changed = 0
    for path in sorted(glob.glob(a.glob)):
        rows = list(csv.reader(open(path, newline="", encoding="utf-8")))
        hit = False
        for r in rows:
            if not r or not r[0].startswith("D3"):
                continue
            want = verdict(float(r[2]), float(r[3]), float(r[4]))
            if r[7].strip() != want:
                print(f"  {os.path.basename(path)}: {float(r[2]):+.4f} "
                      f"[{float(r[3]):+.4f}, {float(r[4]):+.4f}]  "
                      f"{r[7].strip()!r} -> {want!r}")
                r[7] = want
                hit = True
                changed += 1
        if hit and not a.check:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh, lineterminator="\r\n").writerows(rows)
    print(f"{changed} row(s) {'would be' if a.check else ''} relabelled; no number was touched")


if __name__ == "__main__":
    main()
